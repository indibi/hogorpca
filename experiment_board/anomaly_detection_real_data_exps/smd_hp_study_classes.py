import os, sys
from pathlib import Path
import warnings
# import logging
# import datetime
from pprint import pprint
# from copy import deepcopy
# from collections import defaultdict
# from tempfile import TemporaryFile

import numpy as np
import pandas as pd
import torch
# import scipy as sp
from scipy.stats import norm
# from sklearn.neighbors import kneighbors_graph
# from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, precision_recall_curve, auc
import networkx as nx
import optuna
from tqdm import tqdm
import matplotlib.pyplot as plt
import yaml
# from optuna.artifacts import FileSystemArtifactStore
# from optuna.artifacts import upload_artifact
# from optuna.artifacts import download_artifact


BASE_DIR = Path(__file__).resolve().parent.parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / 'results'
CFG_DIR = SCRIPT_DIR / 'configs'

sys.path.append(BASE_DIR.as_posix())

from data.server_machine_dataset import SMDMachineChannel
from data.nyc_taxi_dataset import NYCTaxiDataset
from src.models.lr_ssd.snn__logn_gtv import SNN__LOGN_GTV
from src.multilinear_ops.matricize import matricize
from src.multilinear_ops.tensorize import tensorize
from src.stats.degrees_of_freedom import naive_generalized_lasso_df as calculate_df_naive
from src.study.sample_from_N_simplex import sample_from_N_simplex
from src.study.optuna_constraints_func import initialize_multiple_constraint_functions

# Override optunas default logging level to ERROR only.
# This is to prevent optuna from printing too many logs during the hyperparameter search.
optuna.logging.set_verbosity(optuna.logging.ERROR)


try:
    variance_estimates = pd.read_csv(RESULTS_DIR / 'rank_and_variance_estimates.csv', index_col=[0,1,2]
                                    ).loc['R2']['variance_estimate'].to_dict()
    nyc_variance_estimate = pd.read_csv(RESULTS_DIR / 'nyc_taxi_rank_and_variance_estimates.csv',
                                    index_col='Method').loc['centered_R2']['variance_estimate']
except FileNotFoundError:
    df_rank_and_variance_est = None

class StudyWrapper:
    def __init__(self, model_name, machine_id, channel_id, metric_name,
                cfg_customization={}, seed=0):
        """Initialize the model study on the specified machine and channel.
        
        Loads the study configuration and model configuration from the YAML files
        and sets up the data variables for the specified machine and channel.
        Args:
            model_name (str): Name of the model to be used for the study.
            machine_id (int): ID of the machine to run the study on.
            channel_id (int): ID of the channel to run the study on.
            metric_name (str): Name of the metric to be optimized during the study.
            cfg_customization (dict): Customization for the model and study configurations.
                this dictionary is a nested dictionary where the first key is either 'model' or 'study'
                and the second key is the path to the configuration parameter to be customized.
                an example of a customization dictionary is:
                {'model' : {'max_iter': 2000},
                 'study' : {'n_trials': 100}
                }
        """
        self.data_variables = {'machine_id': machine_id,
                                'channel_id': channel_id}
        with open(CFG_DIR / 'study_configs.yaml') as f:
            study_configs = yaml.safe_load(f)
            if metric_name not in study_configs.keys():
                raise ValueError(f"Metric {metric_name} not found in study_configs.yaml.")
            self.study_config = study_configs[metric_name]
        with open(CFG_DIR / 'model_configs.yaml') as f:
            model_configs = yaml.safe_load(f)
            if model_name not in model_configs:
                raise ValueError(f"Model {model_name} not found in model_configs.yaml.")
            self.model_config = model_configs[model_name]

        study_name = self.model_config['name']
        study_name += f"_m{machine_id}_ch{channel_id}"
        model_customization = cfg_customization.get('model', {})
        study_customization = cfg_customization.get('study', {})
        for key, value in model_customization.items():
            key_path = key.split('.')
            self.model_config = set_nested_value(self.model_config, key_path, value)
            if key_path[-1] != 'device':
                study_name += f"_{key.split('.')[-1]}_{value}"
        for key, value in study_customization.items():
            key_path = key.split('.')
            self.study_config = set_nested_value(self.study_config, key_path, value)
            if key_path[-1] != 'overwrite':
                study_name += f"_{key.split('.')[-1]}_{value}"
        self.study_name = study_name + f"_scfg{metric_name}"
        self.storage = f'sqlite:///{RESULTS_DIR.absolute()}/{model_name}_m{machine_id}_ch{channel_id}.db'
        self.objective_function = SMD_AD_Objective(self.data_variables,
                                        self.study_config,
                                        self.model_config,
                                        self.study_name)
        self.study = self._load_create_or_overwrite_study()
        
    def run_study(self, n_trials=None, n_jobs=1, timeout=None, append_or_complete='append', device=None):
        """Run the study with the specified number of trials and jobs.
        
        Args:
            n_trials (int): Number of trials to run in the study. If None, uses the value from the study configuration.
            n_jobs (int): Number of parallel jobs to run. Defaults to 1.
            timeout (int): Maximum time in seconds to run the study. If None, runs until all trials are completed.
            append_or_complete (str): Whether to append to the existing study or complete it to have n_trials.
                Defaults to 'append'.
            device (str): Device to run the study on. If None, uses the device specified in the model configuration.
        """
        if n_trials is None:
            n_trials = self.study_config.get('n_trials', 100)
        if timeout is None:
            timeout = self.study_config.get('timeout', None)
        if device is not None:
            self.model_config['device'] = device
            self.objective_function.model_select['device'] = device
        
        n_trials = n_trials if append_or_complete == 'append' else n_trials - len(self.study.trials)
        self.study.set_user_attr("study_name", self.study_name)
        # pprint(f"Study configuration for {self.study_name}:")
        # pprint({'model': self.model_config,
        #         'study': self.study_config,
        #         'data': self.data_variables})
        self.study.set_user_attr("config", {'model': self.model_config,
                                            'study': self.study_config,
                                            'data': self.data_variables})
        if n_trials <= 0:
            print(f"Study {self.study_name} already has {len(self.study.trials)} trials. No new trials to run.")
            return
        print(f"Running study {self.study_name} with {n_trials} trials and {n_jobs} jobs.")

        # self.study.set_metric_names([self.study_config['metric']])
        if self.study_config.get('manual', False):
            self._manual_study_run(n_trials=n_trials, n_jobs=n_jobs, timeout=timeout)
        else:
            self.study.optimize(self.objective_function, n_trials=n_trials, n_jobs=n_jobs, timeout=timeout,
                            callbacks=[self.objective_function.callback])
        
        best_result = {}
        best_result['study_name'] = self.study_name
        best_result['results'] = self.study.best_trial.user_attrs
        best_result['params'] = self.study.best_trial.params
        best_result['best_trial'] = self.study.best_trial.number
        best_result['best_value'] = self.study.best_value
        return best_result

    def _manual_study_run(self, n_trials, n_jobs, timeout):
        
        # First phase
        scheme = self.study_config.get('scheme', 'simple_two_phase')
        if scheme == 'simple_two_phase':
            phase_1_cfg = self.study_config['phase_1']
            
            p1_gtvs = phase_1_cfg.get('lda_gtvs',
                                        [1e-7]*len(self.model_config['gtvr_config'])
                                        )[:len(self.model_config['gtvr_config'])]

            t_min = phase_1_cfg['t_min']
            t_max = phase_1_cfg['t_max']
            num_step = phase_1_cfg.get('num_step', 50)
            t_range, t_delta = np.linspace(t_min, t_max, num_step, retstep=True)
            for i, t in enumerate(t_range):
                hps = {f'psi_{j+1}': (1-t) for j in range(len(self.model_config['lr_modes']))}
                hps['lda'] = t
                hps.update({f'lda_gtv_{j+1}': p1_gtvs[j] for j in range(len(p1_gtvs))})
                
                self.study.enqueue_trial(hps, skip_if_exists=True)
            
            self.study.optimize(self.objective_function, n_trials=num_step, n_jobs=n_jobs, timeout=timeout,
                            callbacks=[self.objective_function.callback],
                            show_progress_bar=True)
            # Second phase
            phase_2_cfg = self.study_config['phase_2']
            best_params = self.study.best_params
            best_t = best_params['lda']
            t_range = np.linspace(max([best_t-t_delta,0]),
                                  min([best_t+t_delta,1.0]),
                                  phase_2_cfg.get('t_num_step', 5))
            p2_lda_gtv_min = phase_2_cfg.get('lda_gtv_min',
                                        [1e-8]*len(self.model_config['gtvr_config'])
                                        )[:len(self.model_config['gtvr_config'])]
            p2_lda_gtv_max = phase_2_cfg.get('lda_gtv_max',
                                        [1e-4]*len(self.model_config['gtvr_config'])
                                        )[:len(self.model_config['gtvr_config'])]

            lda_gtv_ranges = [np.geomspace(p2_lda_gtv_min[i],
                                        p2_lda_gtv_max[i],
                                        phase_2_cfg.get('lda_gtv_num_step', 5)[i])
                                        for i in range(len(self.model_config['gtvr_config']))]
            
            for t, lda_gtvs in zip(t_range, 
                                        np.meshgrid(*lda_gtv_ranges, indexing='ij')):
                hps = {f'psi_{j+1}': (1-t) for j in range(len(self.model_config['lr_modes']))}
                hps['lda'] = t
                hps.update({f'lda_gtv_{j+1}': lda_gtvs[j] for j in range(len(lda_gtvs))})
                
                self.study.enqueue_trial(hps, skip_if_exists=True)
            self.study.optimize(self.objective_function, n_trials=len(t_range)*np.prod([len(r) for r in lda_gtv_ranges]),
                                n_jobs=n_jobs, timeout=timeout,
                                callbacks=[self.objective_function.callback],
                                show_progress_bar=True)
        
        if scheme == 'conditional_two_phase':
            phase_1_cfg = self.study_config['phase_1']
            
            p1_gtvs = phase_1_cfg.get('lda_gtvs',
                                        [1e-7]*len(self.model_config['gtvr_config'])
                                        )[:len(self.model_config['gtvr_config'])]

            t_min = phase_1_cfg['t_min']; t_max = phase_1_cfg['t_max']
            num_step = phase_1_cfg.get('num_step', 50)
            t_range, t_delta = np.linspace(t_min, t_max, num_step, retstep=True)
            for i, t in enumerate(t_range):
                hps = {f'psi_{j+1}': (1-t) for j in range(len(self.model_config['lr_modes']))}
                hps['lda'] = t
                hps.update({f'lda_gtv_{j+1}': p1_gtvs[j] for j in range(len(p1_gtvs))})
                
                self.study.enqueue_trial(hps, skip_if_exists=True)
            
            self.study.optimize(self.objective_function, n_trials=num_step, n_jobs=n_jobs, timeout=timeout,
                            callbacks=[self.objective_function.callback],
                            show_progress_bar=True)

            # Filter out trials that violate the condition
            phase_2_cfg = self.study_config['phase_2']
            condition_query = phase_2_cfg.get('condition_query', 'non_zero_S_ratio >= 0.03')

            df_trials = self.study.trials_dataframe()
            df_trials = df_trials.rename(columns=lambda x: x.replace('user_attrs_', '')
                            ).rename(columns=lambda x: x.replace('params_', '')
                            ).rename(columns=lambda x: x.replace('raw_', ''))
            
            df_trials_filtered = df_trials.query(condition_query)
            if len(df_trials_filtered) > 0:
                df_trials = df_trials_filtered
            
            study_df_sorted = df_trials.sort_values(by='value',
                                            ascending=True if self.study_config['direction']=='minimize'
                                                                        else False)

            
            best_t = study_df_sorted.iloc[0]['lda']
            t_range = np.linspace(max([best_t-t_delta,0]),
                                  min([best_t+t_delta,1.0]),
                                  phase_2_cfg.get('t_num_step', 5))
            p2_lda_gtv_min = phase_2_cfg.get('lda_gtv_min',
                                        [1e-8]*len(self.model_config['gtvr_config'])
                                        )[:len(self.model_config['gtvr_config'])]
            p2_lda_gtv_max = phase_2_cfg.get('lda_gtv_max',
                                        [1e-4]*len(self.model_config['gtvr_config'])
                                        )[:len(self.model_config['gtvr_config'])]

            lda_gtv_ranges = [np.geomspace(p2_lda_gtv_min[i],
                                        p2_lda_gtv_max[i],
                                        phase_2_cfg.get('lda_gtv_num_step', 5)[i])
                                        for i in range(len(self.model_config['gtvr_config']))]
            
            for t, lda_gtvs in zip(t_range, 
                                        np.meshgrid(*lda_gtv_ranges, indexing='ij')):
                hps = {f'psi_{j+1}': (1-t) for j in range(len(self.model_config['lr_modes']))}
                hps['lda'] = t
                hps.update({f'lda_gtv_{j+1}': lda_gtvs[j] for j in range(len(lda_gtvs))})
                
                self.study.enqueue_trial(hps, skip_if_exists=True)
            self.study.optimize(self.objective_function, n_trials=len(t_range)*np.prod([len(r) for r in lda_gtv_ranges]),
                                n_jobs=n_jobs, timeout=timeout,
                                callbacks=[self.objective_function.callback],
                                show_progress_bar=True)
        
        if scheme == 'anchored_two_phase':
            phase_1_cfg = self.study_config['phase_1']
            
            p1_gtvs = phase_1_cfg.get('lda_gtvs',
                                        [1e-7]*len(self.model_config['gtvr_config'])
                                        )[:len(self.model_config['gtvr_config'])]

            m_min = phase_1_cfg['m_min']; m_max = phase_1_cfg['m_max']
            num_step = phase_1_cfg.get('num_step', 50)
            m_range = np.logspace(m_min, m_max, num_step)
            lda_anchor = phase_1_cfg.get('lda_anchor', 4.817e-5)

            hps= {'psi': 1.0}
            for i, m in enumerate(m_range):
                
                hps['lda'] = m*lda_anchor
                hps.update({f'lda_gtv_{j+1}': p1_gtvs[j] for j in range(len(p1_gtvs))})
                
                self.study.enqueue_trial(hps, skip_if_exists=True)
            
            self.study.optimize(self.objective_function, n_trials=num_step, n_jobs=n_jobs, timeout=timeout,
                            callbacks=[self.objective_function.callback],
                            show_progress_bar=True)

            # Filter out trials that violate the condition
            phase_2_cfg = self.study_config['phase_2']
            condition_query = phase_2_cfg.get('condition_query', 'non_zero_S_ratio >= 0.03')

            df_trials = self.study.trials_dataframe()
            df_trials = df_trials.rename(columns=lambda x: x.replace('user_attrs_', '')
                            ).rename(columns=lambda x: x.replace('params_', '')
                            ).rename(columns=lambda x: x.replace('raw_', ''))
            
            df_trials_filtered = df_trials.query(condition_query)
            if len(df_trials_filtered) > 0:
                df_trials = df_trials_filtered
            
            study_df_sorted = df_trials.sort_values(by='value',
                                            ascending=True if self.study_config['direction']=='minimize'
                                                                        else False)

            
            best_lda = study_df_sorted.iloc[0]['lda']
            # t_range = np.linspace(max([best_t-t_delta,0]),
            #                       min([best_t+t_delta,1.0]),
            #                       phase_2_cfg.get('t_num_step', 5))
            p2_lda_gtv_min = phase_2_cfg.get('lda_gtv_min',
                                        [1e-8]*len(self.model_config['gtvr_config'])
                                        )[:len(self.model_config['gtvr_config'])]
            p2_lda_gtv_max = phase_2_cfg.get('lda_gtv_max',
                                        [1e-4]*len(self.model_config['gtvr_config'])
                                        )[:len(self.model_config['gtvr_config'])]

            lda_gtv_ranges = [np.geomspace(p2_lda_gtv_min[i],
                                        p2_lda_gtv_max[i],
                                        phase_2_cfg.get('lda_gtv_num_step', 5)[i])
                                        for i in range(len(self.model_config['gtvr_config']))]
            
            hps = {'psi': 1.0 }
            hps['lda'] = best_lda
            for lda_gtvs in zip(np.meshgrid(*lda_gtv_ranges, indexing='ij')):
                hps.update({f'lda_gtv_{j+1}': lda_gtvs[j] for j in range(len(lda_gtvs))})
                
                self.study.enqueue_trial(hps, skip_if_exists=True)
            self.study.optimize(self.objective_function, n_trials=len(t_range)*np.prod([len(r) for r in lda_gtv_ranges]),
                                n_jobs=n_jobs, timeout=timeout,
                                callbacks=[self.objective_function.callback],
                                show_progress_bar=True)



    def _load_create_or_overwrite_study(self):
        """Load or create a new study with the specified name and storage."""
        
        if self.study_config.get('overwrite', False):
            study_names = optuna.get_all_study_summaries(storage=self.storage)
            if self.study_name in study_names:
                optuna.delete_study(study_name=self.study_name, storage=self.storage)
        sampler = self.study_config.get('sampler', 'TPESampler')
        sampler_cfg = self.study_config.get('sampler_cfg', {
            'seed': 0
        })
        if sampler == 'TPESampler':
            self.sampler = optuna.samplers.TPESampler(**sampler_cfg)
        elif sampler == 'RandomSampler':
            self.sampler = optuna.samplers.RandomSampler(**sampler_cfg)
        elif sampler == 'CmaEsSampler':
            self.sampler = optuna.samplers.CmaEsSampler(**sampler_cfg)
        elif sampler == 'GPSampler':
            ## See optuna_constraints_func.py for details on constraint functions
            constraints_func_cfg = sampler_cfg.pop('constraints_func_cfg', None)
            if constraints_func_cfg is not None:
                constraints_func = initialize_multiple_constraint_functions(**constraints_func_cfg)
            else:
                constraints_func = None
            self.sampler = optuna.samplers.GPSampler(**sampler_cfg, constraints_func=constraints_func)
        
        if isinstance(self.study_config.get('metric'), str):
            study = optuna.create_study(study_name=self.study_name,
                                            sampler=self.sampler, # optuna_sampler,#
                                            storage=self.storage,
                                            load_if_exists=True,
                                            direction=self.study_config['direction'])
        elif isinstance(self.study_config.get('metric'), list):
            study = optuna.create_study(study_name=self.study_name,
                                            sampler=self.sampler, # optuna_sampler,
                                            storage=self.storage,
                                            load_if_exists=True,
                                            directions=self.study_config['directions'])
        else:
            raise ValueError("Study metric must be a string or a list of strings.")
        return study

class StudyCallback:
    def __init__(self, artifact_folder_path, artifact_prefix):
        """Callback class to save the best trial's artifacts and study artifacts.

        This class is used to save the best trial's artifacts and study artifacts to a specified folder.
        The artifact data is passed to the callback instance within the objective function to the
        `.trial_artifacts` and `.study_artifacts` class instance variables are saved in a `.npz` file format.
        Args:
            artifact_folder_path (Path or str): Path to the folder where artifacts will be saved.
            artifact_prefix (str): Prefix for the artifact file names.
        """
        self.trial_artifacts = None
        self.study_artifacts = None
        # self.artifact_store = FileSystemArtifactStore(artifact_path)
        self.artifact_folder_path = artifact_folder_path
        self.artifact_prefix = artifact_prefix
        os.makedirs(artifact_folder_path, exist_ok=True)

    def __call__(self, study, frozen_trial):
        winner = study.user_attrs.get("winner", None)
        try:
            if study.best_value and winner != study.best_value:
                study.set_user_attr("winner", study.best_value)
                if winner:
                    print(f"Trial {frozen_trial.number} achieved value: {frozen_trial.value} with params:")
                    pprint(frozen_trial.params)
                    print("--"*20)
                    pprint(frozen_trial.user_attrs)
                else:
                    print(f"Initial trial {frozen_trial.number} achieved value: {frozen_trial.value}")
                
                if self.study_artifacts is not None:
                    # Save the best trial's data to an artifact for later analysis
                    file_path = Path(self.artifact_folder_path) / (self.artifact_prefix + "_sa.npz")
                    with open(file_path, 'wb') as f:
                        np.savez(f, **self.study_artifacts)
                    if study.user_attrs.get("artifact_path", None) is None:
                        study.set_user_attr("artifact_path", str(file_path))
                    self.study_artifacts = None
                if self.trial_artifacts is not None:
                    # Save the best trial's data to an artifact for later analysis
                    file_path = Path(self.artifact_folder_path) / (self.artifact_prefix + f"_ta_{frozen_trial.number}.npz")
                    with open(file_path, 'wb') as f:
                        np.savez(f, **self.trial_artifacts)
                    if frozen_trial.user_attrs.get("artifact_path", None) is None:
                        frozen_trial.set_user_attr("artifact_path", str(file_path))
                    self.trial_artifacts = None
        except ValueError:
            print(f"Trial {frozen_trial.number} achieved value: {frozen_trial.value} with params:")
            pprint(frozen_trial.params)
            print("--"*20)
            pprint(frozen_trial.user_attrs)


class SMD_AD_Objective:
    def __init__(self, data_variables, study_config, model_config, study_name):
        # Load Server Machine Data for machine and channel
        if data_variables['machine_id'] == 0:
            spatial_edge_weighting = study_config.get('spatial_edge_weighting', {})
            self.sm_data = NYCTaxiDataset(edge_weighting=spatial_edge_weighting)
            self.data_variables = data_variables
            
            self.data_variables['mode_nnorms'] = [380265.44, 346187.32, 260182.74, 339817.44] # Mode Nuclear Norms
            self.data_variables['mode_nn_weights'] = [1.0, 1.098, 1.462, 1.119]         # Mode Nuclear Norm Weights
            self.data_variables['mode_sval_concentrations'] = [1.0, 0.961, 0.591, 0.802]# Concentration Weights
            # Normalized Concentration Weights
            self.data_variables['normalized_sval_concentration_weights'] = [1.0, 1.188, 2.01, 1.473]
            if 'estimated_variance' not in data_variables: 
                self.data_variables['estimated_variance'] = 78.37# 1e-7# 0.001 # 5.0 # 0.01 # 0.1 # 5.0 # 20.0 # nyc_variance_estimate
            else:                                          #ch_17# ch_16# ch_14 #ch_15#ch_13 #ch_12#ch_11# ch_9 #
                self.data_variables['estimated_variance'] = 78.37# 1e-7# 0.001 # 5.0 # 0.01 # 0.1 # 5.0 # 20.0 # 48.18 #78.37
            
        else:
            self.sm_data = SMDMachineChannel(data_variables['machine_id'], data_variables['channel_id'])
            self.data_variables = data_variables

            if variance_estimates is not None:
                self.data_variables['estimated_variance']= variance_estimates[(data_variables['machine_id'], data_variables['channel_id'])]
            else:
                self.data_variables['estimated_variance'] = self.sm_data.estimate_noise_variance()
        self.study_config = study_config
        self.model_config = model_config
        
        self.model_select = { # Model selection based on the model_config
            'graph': self.model_config['graph'],
            'lr_modes': self.model_config['lr_modes'],
            'graph_modes': self.model_config['graph_modes'],
            'grouping': self.model_config['grouping'],
            'weighing': self.model_config['weighing'],
            'r_hop': self.model_config['r_hop'],
            'soft_constrained': self.model_config.get('soft_constrained', False),
            'device': self.model_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'),
            'dtype': self.model_config.get('dtype', torch.float64),
            'verbose': self.model_config.get('verbose', 0),
            'gtvr_config': self.model_config['gtvr_config'],
            }
        self.model_run = {
            'max_iter': self.model_config.get('max_iter', 5000),
            'rho': self.model_config.get('rho', 0.03),
            'err_tol': self.model_config.get('err_tol', 1e-4),
            'rho_update': self.model_config.get('rho_update', 'domain_parametrization'),
            'rho_update_thr': self.model_config.get('rho_update_thr', 100)
        }

        ARTIFACT_DIR = RESULTS_DIR / 'artifacts'
        self.callback = StudyCallback(ARTIFACT_DIR, artifact_prefix=study_name)
        
        self.study_config['estimated_variance'] = self.data_variables['estimated_variance']
        if self.data_variables.get('mode_nnorms', None) is not None:
            self.study_config['mode_nnorms'] = self.data_variables['mode_nnorms']
            self.study_config['mode_nn_weights'] = self.data_variables['mode_nn_weights']
            self.study_config['mode_sval_concentrations'] = self.data_variables['mode_sval_concentrations']
            self.study_config['normalized_sval_concentration_weights'] = self.data_variables['normalized_sval_concentration_weights']


    def __call__(self, trial):
        # torch.set_grad_enabled(False)
        # Ask for hyper-parameters from the trial and run the model
        hyper_parameters = self._ask_hyperparameters(trial)
        G = self.sm_data.G
        if self.model_select['graph'] == 'temporal':
            G = self.sm_data.Gt
        elif self.model_select['graph'] == 'spatio-temporal':
            Gt = self.sm_data.Gt
            pd = self.model_select.get('graph_product', 'cartesian')
            if pd == 'cartesian':
                G = nx.cartesian_product(G, Gt)
            elif pd == 'strong':
                G = nx.strong_product(G, Gt)
        model = SNN__LOGN_GTV(self.sm_data.Y, G, **self.model_select)
        X, S = model(hyper_parameters['psis'], hyper_parameters['lda'],
                      hyper_parameters['lda_gtvs'], lda_f=hyper_parameters['lda_f'],
                        **self.model_run)
        
        # Calculate the degrees of freedom & other metrics for hp-search
        results = self._degrees_of_freedom(model)
        diff = len(model.Y.ravel()) - results['dof']
        error_norm = torch.linalg.norm(model.Y - X - S).item()
        results['loss'] = error_norm**2
        results['iterations'] = model.it
        variance = self.data_variables.get('estimated_variance', error_norm**2/model.Y.numel())
        dof = results['dof']
        dof_X = results['dof_X']
        dof_S = results['dof_S']
        p = results['num_covariates']
        n = model.Y.numel()
        p_S = p - n
        p_X = p - p_S
        lda_f = hyper_parameters['lda_f']
        # if lda_f ==0:
        #     results['nll'] = 0.0
        # else:
        #     results['nll'] = 0.5 * (lda_f * results['loss'] - np.log(0.5*lda_f/np.pi))
        results['nll'] = 0.5 * ((1/variance) * results['loss'] - np.log(0.5/(np.pi*variance)))
        results['bic'] = 2*results['nll'] + np.log(results['num_covariates']) * results['dof']
        results['aic'] = 2*results['nll'] + 2*results['dof']
        results['gcv'] = n*(error_norm/max(1, diff))**2
        results['gic_1'] = error_norm**2 + dof*variance*np.log(n)
        results['gic_2'] = error_norm**2 + dof*variance*p**(1/3)
        results['gic_3'] = error_norm**2 + variance*2*(dof_X*np.log(p_X) + dof_S*np.log(p_S))#dof*variance*2*np.log(p)
        results['gic_4'] = error_norm**2 + dof*variance*2*(np.log(p)+np.log(np.log(p)))
        results['gic_5'] = error_norm**2 + dof*variance*(np.log(p)*np.log(np.log(n)))
        results['gic_6'] = error_norm**2 + dof*variance*(np.log(p)*np.log(n))

        # results.update(model.calculate_concentrations())
        # Calculate anomaly detection and scoring performance metrics
        if isinstance(self.sm_data, SMDMachineChannel):
            scoring_scores = self.sm_data.anomaly_scoring_score(model.S.abs())
        elif isinstance(self.sm_data, NYCTaxiDataset):
            if self.study_config.get('anomaly_scoring', 'abs_S') == 'abs_S':
                scoring_scores = self.sm_data.anomaly_scoring_score(model.S.abs().cpu().numpy())
            elif self.study_config.get('anomaly_scoring', 'abs_S') == 'residual':
                scoring_scores = self.sm_data.anomaly_scoring_score(torch.abs(model.Y - model.X).cpu().numpy())
        

        non_zeros = ~torch.isclose( torch.zeros_like(model.S), model.S)
        results['non_zero_S_ratio'] = non_zeros.sum().item()/non_zeros.numel()

        if non_zeros.sum() <10:
            t_labels = non_zeros.cpu().numpy()
        else:
            non_zero_S = S[non_zeros].cpu().numpy()
            var_brac = np.quantile(np.abs(non_zero_S), 0.1)
            var = np.var(non_zero_S[np.abs(non_zero_S) < var_brac].ravel())
            t_labels = np.abs(S.cpu().numpy()) > np.sqrt(var_brac)
        
        # results['non_zero_S_ratio'] = t_labels.sum().item()/t_labels.size
        if isinstance(self.sm_data, SMDMachineChannel):
            detection_scores = self.sm_data.anomaly_detection_score(t_labels)

            results['au_prc'] = scoring_scores['au_prc']
            results['au_roc'] = scoring_scores['au_roc']
            for key, value in detection_scores.items():
                results['raw_'+key] = value
            
            
            likelihood = convert_to_likelihood(S, self.sm_data.G)
            scoring_scores = self.sm_data.anomaly_scoring_score(likelihood)
            results['likelihood_au_prc'] = scoring_scores['au_prc']
            results['likelihood_au_roc'] = scoring_scores['au_roc']
        elif isinstance(self.sm_data, NYCTaxiDataset):
            detection_scores = self.sm_data.anomaly_detection_score(t_labels)
            for key, value in detection_scores.items():
                results[key] = value
            for key, value in scoring_scores.items():
                results['scoring_'+key] = value
            likelihood = None

        # Pass the resulting tensor to the callback class instance
        if self.callback is not None:
            self.callback.study_artifacts = {
                'likelihood': likelihood,
                'S': S.cpu().numpy(),
            }
        # Set the user attributes for the trial
        for key, value in results.items():
            trial.set_user_attr(key, value)
        return results[self.study_config['metric']]

    def _degrees_of_freedom(self, model):
        """Estimate the degrees of freedom for the model."""
        num_parameter = model.num_parameters(rtol=0.999, abs_thr=1e-20,
                                    zero_tensor_thr_factor=None,
                                    force_core_thresholding=True)
        # For LOGN regularization without the variance regularization,
        # if self.model_select['grouping'] == 'neighbor' and self.model_select['r_hop'] ==0:
        # l1 regularization
        if len(self.model_select['gtvr_config'])==0:
            # HoRPCA
            dof_S = num_parameter['S']
        elif len(self.model_select['gtvr_config'])==1:
            # Anisotropic GTV 
            dof_S = single_fused_df(model)
        elif len(self.model_select['gtvr_config']) == 2:
            # Double fused anisotropic GTV
            dof_S = double_fused_df(model)
        else:
            raise NotImplementedError(f"Degrees of freedom calculation for a model with {len(self.model_select['gtvr_config'])}" +
                             "combined GTV regularizations is not implemented yet.")
        # else:
        #     # LOGN regularization
        #     warnings.warn("Degrees of freedom calculation for LOGN regularization is not implemented yet. Using number of non-zeros as an estimate.")
        #     dof_S = num_parameter['S']
        
        return {'dof_X': num_parameter['X'],
                'dof_S': dof_S,
                'dof': num_parameter['X'] + dof_S,
                'num_covariates': num_parameter['P'],
                'truncation_r2_gcv': num_parameter['X_gcv_r2_value'],
                'truncation_r2': num_parameter['X_r2_value'],
                'X_t_rank': num_parameter['X_t_rank']
                }


    def _ask_hyperparameters(self, trial):
        hyperparameters = self.model_config.get('hyperparameters', {})
        soft_constrained = self.model_select['soft_constrained']
        sample_simplex = self.study_config.get('sample_simplex', False)
        if sample_simplex:
            N = len(self.model_select['lr_modes']) + 1
            if soft_constrained:
                ts = list(sample_from_N_simplex(trial, N+1).ravel())
                lda_f = ts[0]
                ts = ts[1:]
            else:
                ts = list(sample_from_N_simplex(trial, N).ravel())
                lda_f = hyperparameters.get('lda_f', 1/self.study_config.get('estimated_variance', 1.0))
            lda = ts[0]
            psis = ts[1:]
            lda_gtvs = hyperparameters.get('lda_gtvs', None)
        else:
            psis = hyperparameters.get('psis', None)
            lda = hyperparameters.get('lda', None)
            lda_gtvs = hyperparameters.get('lda_gtvs', None)
            psi_weighting = self.study_config.get('psi_weighting', None)
            if psi_weighting != None:
                psis = np.array(self.study_config[psi_weighting]) # normalized_sval_concentration_weights
                psi = trial.suggest_float("psi",
                                    self.study_config.get('psi_min', 0.01),
                                    self.study_config.get('psi_max', 20),
                                    log=self.study_config.get('psi_log', True))
                psis = psi*psis
            if psis is None:
                psis = [trial.suggest_float(f"psi_{i}",
                                    self.study_config.get('psi_min', 0.01),
                                    self.study_config.get('psi_max', 20),
                                    log=self.study_config.get('psi_log', True))
                                    for i in range(1, len(self.model_select['lr_modes']) + 1)]
            if lda is None:
                lda = trial.suggest_float("lda",
                                    self.study_config.get('lda_min', 0.001),
                                    self.study_config.get('lda_max', 20),
                                    log=self.study_config.get('lda_log', True))
            if soft_constrained:
                lda_f = hyperparameters.get('lda_f', 1/self.study_config.get('estimated_variance', None))
                if lda_f is None:
                    lda_f = trial.suggest_float("lda_f",
                                self.study_config.get('lda_f_min', 0.001),
                                self.study_config.get('lda_f_max', 20),
                                log=self.study_config.get('lda_f_log', True))
            else:
                lda_f = 0
        if lda_gtvs is None:
            lda_gtvs = [trial.suggest_float(f"lda_gtv_{i}",
                            self.study_config.get(f'lda_gtv_{i}_min',
                                    self.study_config.get(f'lda_gtv_min', 1e-7)),
                            self.study_config.get(f'lda_gtv_{i}_max',
                                    self.study_config.get(f'lda_gtv_max', 10)),
                            log=self.study_config.get('lda_gtv_log', True))
                            for i in range(1, len(self.model_select['gtvr_config']) + 1)]
        hyperparameters = {'psis': psis,
                            'lda': lda,
                            'lda_gtvs': lda_gtvs,
                            'lda_f': lda_f}
        return hyperparameters

def set_nested_value(data, keys, value):
    """
    Sets a value in a nested dictionary given a list of keys.

    Args:
        data (dict): The dictionary to modify.
        keys (list): A list of keys representing the path to the value.
        value: The value to set.
    """
    if not keys:
        return value
    if not isinstance(data, dict):
      data = {}
    data[keys[0]] = set_nested_value(data.get(keys[0], {}), keys[1:], value)
    return data

def convert_to_likelihood(S, G, graph_mode=1, radius=2):
    """Fit a Gaussian to the neighbors of each node in the graph and return the likelihood."""
    S_loc = matricize(S, [graph_mode]).cpu().numpy()
    r_A = np.eye(S_loc.shape[0])+np.linalg.matrix_power(nx.adjacency_matrix(G).toarray(),radius)
    likelihood = np.zeros(S_loc.shape)
    for s in range(S_loc.shape[0]):
        mask = r_A[np.where(r_A[s,]!=0),:].astype(bool)
        nbd = S_loc[mask[:,0,:].ravel(),:]
        # Append neighbors from additional columns in mask
        for m in range(1, mask.shape[1]):
            nbd = np.vstack((nbd,S_loc[mask[:,m,:].ravel(),:] ))
                
        W = np.zeros(nbd.shape)
        # Iterate through the columns in steps of block_size
        for i1 in range(0, S_loc.shape[1], 1440):
            # Slice the matrix to get the block (columns from i to i + block_size)
            if i1==0:
                block = nbd[:, i1:i1 + 1440]
                for loc in range(nbd.shape[0]):
                    W[loc, i1:i1+1440] = norm.pdf(np.linalg.norm(block[loc,:] - S_loc[s,i1:i1+1440]),0,30)
            elif i1 == S_loc.shape[1]-1440: #23040:
                block = nbd[:, i1-1440:i1 + 1440]
                for loc in range(nbd.shape[0]):
                    W[loc, i1:i1+1440] = norm.pdf(np.linalg.norm(block[loc,:] - S_loc[s,i1-1440:i1+1440]),0,30)
            else:
                block = nbd[:, i1-1440:i1 + 2880]
                for loc in range(nbd.shape[0]):
                    W[loc, i1:i1+1440] = norm.pdf(np.linalg.norm(block[loc,:] - S_loc[s,i1-1440:i1+2880]),0,30)
                        
        mean = np.sum(W * nbd) / np.sum(W)
        sd = np.sqrt(np.sum(W * (nbd - mean)**2) / np.sum(W))
        if sd == 0:
            sd = sd+(10e-20)
        likelihood[s,] = np.log(sd) + (0.5*np.power(((S_loc[s,] - mean)/sd),2))
    return tensorize(likelihood, S.shape, [graph_mode])


def single_fused_df(model):
    """Estimate degrees of freedom naively for fused lasso model"""
    I = model.ops[0]['I'].to_dense()
    BT = model.ops[0]['B.T'].to_dense()
    D = torch.vstack([BT, I])
    
    # V_sum = model.V.sum(dim=0, keepdim=False).coalesce().to_dense()
    # S_ = tensorize(V_sum, model.Y.shape, model.graph_modes)
    S_ = matricize(model.S, model.vr_config[0]['mode'])
    W_ = BT@S_
    # nonzero_edge = (model.Ws[0].T!=0) 
    # nonzero_node = (S_.T!=0)
    nonzero_edge = ~torch.isclose( torch.zeros_like(W_), W_).T
    nonzero_node = ~torch.isclose( torch.zeros_like(S_), S_).T
    # print(f"nonzero_node shape: {nonzero_node.shape}, nonzero_edge shape: {nonzero_edge.shape}")
    Bindices = torch.hstack([nonzero_edge, nonzero_node])
    # print(f"Bindices shape: {Bindices.shape}")
    
    df = calculate_df_naive(Bindices, D)
    return df.item()

def double_fused_df(model):
    mode1 = model.vr_config[0]['mode']
    mode2 = model.vr_config[1]['mode']
    modes = mode1 + mode2
    assert len(modes) == 2, "Double fused degrees of freedom calculation implementation requires two and only two modes."
    I1 = model.ops[0]['I'].to_dense()
    I2 = model.ops[1]['I'].to_dense()
    BT1 = model.ops[0]['B.T'].to_dense()
    BT2 = model.ops[1]['B.T'].to_dense()
    I = torch.diag(torch.ones(I1.shape[0]*I2.shape[0])).to(device=I1.device)
    D = torch.vstack([torch.kron(BT1, I2),
                      torch.kron(I1, BT2),
                      I])
    
    # V_sum = model.V.sum(dim=0, keepdim=False).coalesce().to_dense()
    # S_ = tensorize(V_sum, model.Y.shape, model.graph_modes)
    S_ = matricize(model.S, modes)
    nonzero_node = torch.isclose( torch.zeros_like(S_), S_).T
    # nonzero_node = (matricize(S_, modes) !=0).T # shape (Batch, V_1 . V_2)

    warnings.warn("Double fused degrees of freedom calculation is naive and may not be accurate."+
                  "Use with caution. single fused degrees of freedom above may be better.")
    w1_dim = list(model.Y.shape)
    w1_dim[mode1[0]-1] = BT1.shape[0]
    Ws = tensorize(model.Ws[0], w1_dim, mode1)
    nonzero_mode1_edge = matricize(Ws, modes).T !=0 # shape (Batch, V_S . V_T)

    w2_dim = list(model.Y.shape)
    w2_dim[mode2[0]-1] = BT2.shape[0]
    Ws = tensorize(model.Ws[1], w2_dim, mode2)
    nonzero_mode2_edge = matricize(Ws, modes).T !=0 # shape (Batch, V_S . V_T)
    
    Bindices = torch.hstack([nonzero_mode1_edge,
                             nonzero_mode2_edge,
                             nonzero_node])
    # print(f"Bindices shape: {Bindices.shape}")
    df = calculate_df_naive(Bindices, D)
    return df.item()