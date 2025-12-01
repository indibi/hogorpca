from pprint import pprint
# from contextlib import nullcontext
from functools import partial

import networkx as nx
import mlflow
import optuna
import torch
import numpy as np
from sklearn.metrics import precision_recall_curve, auc, roc_curve
from dask.distributed import Client, wait
import matplotlib.pyplot as plt
from tqdm import tqdm


from src.study.scientist import ExperimentBase, ExperimentTrialRunnerBase, ModelRunnerBase

from src.models.horpca.horpca_torch import HoRPCA_Singleton
from src.models.lr_ssd.snn_logs import SNN_LOGS

from src.synthetic_data.generate_lr_data import generate_low_rank_data
from src.synthetic_data.generate_anomaly import generate_spatio_temporal_anomaly
from src.multilinear_ops.t2m import t2m

import warnings
warnings.filterwarnings("ignore")


class IdentifiabilityStudyExp1(ExperimentBase):
    def execute(self, sub_experiment_name,
                resolution,
                n_repeat=1,
                seed_start=1,
                devices=[f'cuda:{i}' for i in range(torch.cuda.device_count())]):
        """Execute identifiability experiment 1
        
        This experiment is designed to study the identifiability of the low-rank and sparse decomposition
        """
        # Independent variables
        # independent_vars = {
        # 'model': {'type': 'categorical', 'values': ['HoRPCA', 'SNN_LOGS']},
        # 'spread': {'type': 'categorical', 'values': ['isotropic', 'anisotropic']}, #anomaly_type, # 'isotropic' or 'anisotropic'}
        # 'radius': {'type': 'int', 'low': 0, 'high': 1000000},
        # }
        # runtime_misc = ['seed']
        # horpca_independent_vars = {
        # 't': {'type': 'float', 'low': eps, 'high': 1},
        # }
        # horpca_runtime_misc = ['device']
        ### SNN_LOGS
        # snn_logs_independent_vars = {
            # 't': {'type': 'float', 'low': eps, 'high': 1},
            # 'grouping': {'type': 'categorical', 'values': ['neighbor', 'edge']},
            # 'weighing': {'type': 'categorical', 'values': ['size_normalized', 'size_normalized_inv', 'uniform']},
            # 'r_hop': {'type': 'int', 'low': 1, 'high': 3},
        # }
        # snn_logs_runtime_misc = ['device']
        if sub_experiment_name == '1a':
            return self.execute_exp1a(resolution, n_repeat, seed_start, devices)
        elif sub_experiment_name == '1b':
            return self.execute_exp1b(resolution, n_repeat, seed_start, devices)
        

    def execute_exp1a(self, resolution,
                      n_repeat=1, seed_start=1,
                      devices=[f'cuda:{i}' for i in range(torch.cuda.device_count())]):
        """Execute experiment 1a. 
        
        This experiment is aimed at answering Grouping Question 1.a
        Anomaly spread is Isotropic.
        The ['HoRPCA', 'SNN_LOGS(N1)', 'SNN_LOGS(/N1)', 'SNN_LOGS(-E)'] models are compared.
        """
        
        eps = self.experiment_runner.model_runners['HoRPCA'].model_control_vars['eps']
        t_range = np.linspace(eps, 1, resolution+1, endpoint=True)

        # models = ['HoRPCA', 'SNN_LOGS(N1)', 'SNN_LOGS(/N1)', 'SNN_LOGS(-E)']
        exp_1a_vars ={
            'spread': 'isotropic',
            'r_hop': 1,
            'radius': 2
        }
        exp_1a_misc = {}
        for i in range(n_repeat):
            exp_1a_misc['seed'] = seed_start + i

            exp_1a_vars['model'] = 'HoRPCA'
            for t in t_range:
                exp_1a_vars['t'] = t
                exp_1a_misc['device'] = devices[i % len(devices)]
                self.study.enqueue_trial(exp_1a_vars, exp_1a_misc)
            
            exp_1a_vars['model'] = 'SNN_LOGS(N1)'
            exp_1a_vars['grouping'] = 'neighbor'
            exp_1a_vars['weighing'] = 'size_normalized'
            for t in t_range:
                exp_1a_vars['t'] = t
                exp_1a_misc['device'] = devices[i % len(devices)]
                self.study.enqueue_trial(exp_1a_vars, exp_1a_misc)
            
            exp_1a_vars['model'] = 'SNN_LOGS(/N1)'
            exp_1a_vars['grouping'] = 'neighbor'
            exp_1a_vars['weighing'] = 'size_normalized_inv'
            for t in t_range:
                exp_1a_vars['t'] = t
                exp_1a_misc['device'] = devices[i % len(devices)]
                self.study.enqueue_trial(exp_1a_vars, exp_1a_misc)
            
            exp_1a_vars['model'] = 'SNN_LOGS(-E)'
            exp_1a_vars['grouping'] = 'edge'
            exp_1a_vars['weighing'] = 'uniform'
            for t in t_range:
                exp_1a_vars['t'] = t
                exp_1a_misc['device'] = devices[i % len(devices)]
                self.study.enqueue_trial(exp_1a_vars, exp_1a_misc)
            
            self.study.optimize(self.experiment_runner,
                                n_trials=resolution*4,
                                n_jobs=len(devices),
                                show_progress_bar=True)
        
        df = self.study.trials_dataframe()
        for col_name in df.columns:
            if col_name.startswith('params_'):
                df.rename(columns={col_name:col_name.replace('params_','')}, inplace=True)
            if col_name.startswith('values_'):
                df.rename(columns={col_name:col_name.replace('values_','')}, inplace=True)
            if col_name.startswith('user_attrs_'):
                df.rename(columns={col_name:col_name.replace('user_attrs_','')}, inplace=True)
        exp1a_filter = df['model'].isin(['HoRPCA', 'SNN_LOGS(N1)', 'SNN_LOGS(/N1)', 'SNN_LOGS(-E)'])
        exp1a_filter &= df['spread'] == 'isotropic'
        exp1a_filter &= df['r_hop'] == 1
        exp1a_filter &= df['radius'] == 2
        return df[exp1a_filter]
    
    def execute_exp1b(self, resolution,
                        n_repeat=1, seed_start=1,
                        devices=[f'cuda:{i}' for i in range(torch.cuda.device_count())]):
        
        eps = self.experiment_runner.model_runners['HoRPCA'].model_control_vars['eps']
        t_range = np.linspace(eps, 1, resolution+1, endpoint=True)

        # models = ['HoRPCA', 'SNN_LOGS(N1)', 'SNN_LOGS(/N1)', 'SNN_LOGS(-E)']
        exp_1a_vars ={
            'spread': 'anisotropic',
            'r_hop': 1,
            'radius': 12
        }
        exp_1a_misc = {}
        for i in range(n_repeat):
            exp_1a_misc['seed'] = seed_start + i

            exp_1a_vars['model'] = 'HoRPCA'
            for t in t_range:
                exp_1a_vars['t'] = t
                exp_1a_misc['device'] = devices[i % len(devices)]
                self.study.enqueue_trial(exp_1a_vars, exp_1a_misc)
            
            exp_1a_vars['model'] = 'SNN_LOGS(N1)'
            exp_1a_vars['grouping'] = 'neighbor'
            exp_1a_vars['weighing'] = 'size_normalized'
            for t in t_range:
                exp_1a_vars['t'] = t
                exp_1a_misc['device'] = devices[i % len(devices)]
                self.study.enqueue_trial(exp_1a_vars, exp_1a_misc)
            
            exp_1a_vars['model'] = 'SNN_LOGS(/N1)'
            exp_1a_vars['grouping'] = 'neighbor'
            exp_1a_vars['weighing'] = 'size_normalized_inv'
            for t in t_range:
                exp_1a_vars['t'] = t
                exp_1a_misc['device'] = devices[i % len(devices)]
                self.study.enqueue_trial(exp_1a_vars, exp_1a_misc)
            
            exp_1a_vars['model'] = 'SNN_LOGS(-E)'
            exp_1a_vars['grouping'] = 'edge'
            exp_1a_vars['weighing'] = 'uniform'
            for t in t_range:
                exp_1a_vars['t'] = t
                exp_1a_misc['device'] = devices[i % len(devices)]
                self.study.enqueue_trial(exp_1a_vars, exp_1a_misc)
            
            self.study.optimize(self.experiment_runner,
                                n_trials=resolution*4,
                                n_jobs=len(devices),
                                show_progress_bar=True)
        
        df = self.study.trials_dataframe()
        for col_name in df.columns:
            if col_name.startswith('params_'):
                df.rename(columns={col_name:col_name.replace('params_','')}, inplace=True)
            if col_name.startswith('values_'):
                df.rename(columns={col_name:col_name.replace('values_','')}, inplace=True)
            if col_name.startswith('user_attrs_'):
                df.rename(columns={col_name:col_name.replace('user_attrs_','')}, inplace=True)
        exp1a_filter = df['model'].isin(['HoRPCA', 'SNN_LOGS(N1)', 'SNN_LOGS(/N1)', 'SNN_LOGS(-E)'])
        exp1a_filter &= df['spread'] == 'isotropic'
        exp1a_filter &= df['r_hop'] == 1
        exp1a_filter &= df['radius'] == 12
        return df[exp1a_filter]
    

    def execute_exp1c(self, resolution,
                        n_repeat=1, seed_start=1,
                        devices=[f'cuda:{i}' for i in range(torch.cuda.device_count())]):
        
        eps = self.experiment_runner.model_runners['HoRPCA'].model_control_vars['eps']
        t_range = np.linspace(eps, 1, resolution+1, endpoint=True)

        # models = ['HoRPCA', 'SNN_LOGS(N1)', 'SNN_LOGS(/N1)', 'SNN_LOGS(-E)']
        exp_1a_vars ={
            'spread': 'anisotropic',
            'r_hop': 1,
            'radius': 12
        }
        exp_1a_misc = {}
        for i in range(n_repeat):
            exp_1a_misc['seed'] = seed_start + i

            exp_1a_vars['model'] = 'HoRPCA'
            for t in t_range:
                exp_1a_vars['t'] = t
                exp_1a_misc['device'] = devices[i % len(devices)]
                self.study.enqueue_trial(exp_1a_vars, exp_1a_misc)
            
            exp_1a_vars['model'] = 'SNN_LOGS(N1)'
            exp_1a_vars['grouping'] = 'neighbor'
            exp_1a_vars['weighing'] = 'size_normalized'
            for t in t_range:
                exp_1a_vars['t'] = t
                exp_1a_misc['device'] = devices[i % len(devices)]
                self.study.enqueue_trial(exp_1a_vars, exp_1a_misc)
            
            exp_1a_vars['model'] = 'SNN_LOGS(/N1)'
            exp_1a_vars['grouping'] = 'neighbor'
            exp_1a_vars['weighing'] = 'size_normalized_inv'
            for t in t_range:
                exp_1a_vars['t'] = t
                exp_1a_misc['device'] = devices[i % len(devices)]
                self.study.enqueue_trial(exp_1a_vars, exp_1a_misc)
            
            exp_1a_vars['model'] = 'SNN_LOGS(-E)'
            exp_1a_vars['grouping'] = 'edge'
            exp_1a_vars['weighing'] = 'uniform'
            for t in t_range:
                exp_1a_vars['t'] = t
                exp_1a_misc['device'] = devices[i % len(devices)]
                self.study.enqueue_trial(exp_1a_vars, exp_1a_misc)
            
            self.study.optimize(self.experiment_runner,
                                n_trials=resolution*4,
                                n_jobs=len(devices),
                                show_progress_bar=True)
        
        df = self.study.trials_dataframe()
        for col_name in df.columns:
            if col_name.startswith('params_'):
                df.rename(columns={col_name:col_name.replace('params_','')}, inplace=True)
            if col_name.startswith('values_'):
                df.rename(columns={col_name:col_name.replace('values_','')}, inplace=True)
            if col_name.startswith('user_attrs_'):
                df.rename(columns={col_name:col_name.replace('user_attrs_','')}, inplace=True)
        exp1a_filter = df['model'].isin(['HoRPCA', 'SNN_LOGS(N1)', 'SNN_LOGS(/N1)', 'SNN_LOGS(-E)'])
        exp1a_filter &= df['spread'] == 'isotropic'
        exp1a_filter &= df['r_hop'] == 1
        exp1a_filter &= df['radius'] == 12
        return df[exp1a_filter]

        # Independent variables
        # independent_vars = {
        # 'model': {'type': 'categorical', 'values': ['HoRPCA', 'SNN_LOGS']},
        # 'spread': {'type': 'categorical', 'values': ['isotropic', 'anisotropic']}, #anomaly_type, # 'isotropic' or 'anisotropic'}
        # 'radius': {'type': 'int', 'low': 0, 'high': 1000000},
        # }
        # runtime_misc = ['seed']
        # horpca_independent_vars = {
        # 't': {'type': 'float', 'low': eps, 'high': 1},
        # }
        # horpca_runtime_misc = ['device']
        ### SNN_LOGS
        # snn_logs_independent_vars = {
            # 't': {'type': 'float', 'low': eps, 'high': 1},
            # 'grouping': {'type': 'categorical', 'values': ['neighbor', 'edge']},
            # 'weighing': {'type': 'categorical', 'values': ['size_normalized', 'size_normalized_inv', 'uniform']},
            # 'r_hop': {'type': 'int', 'low': 1, 'high': 3},
        # }
        # snn_logs_runtime_misc = ['device']
    
    @property
    def directions(self):
        return ['minimize', 'minimize', 'minimize', 'minimize', 'maximize', 'maximize']

def experiment_1_runners(dim, ranks, 
                               graph_type, graph_dim,
                               NoA, local_dist, time_m, local_m,
                               duration, amplitude,
                               eps,
                               lr_modes, err_tol, maxit, rho_upd, rho_upd_thr,
                               graph_modes
                               ):
    """Parse experiment settings and return the necessary runners.
    Args:
        dim (tuple of int): tensor dimensions
        ranks (tuple of int): _description_
        graph_type (str): 'grid' ####, 'Gnm', 'random_regular'
        graph_dim (tuple): dimension of the grid graph
        NoA (int): number of anomalies
        local_dist (_type_): _description_
        time_m (int): index (starts from 1) of the mode to be treated as temporal
        local_m (int): index (starts from 1) of the mode to be treated as spatial
        duration (int): temporal duration of anomaly
        amplitude (float): anomaly amplitude
        eps (float): small value for the differentiation of the model outputs for id study
        lr_modes (list of ints): list of integers representing the modes of the tensor to be treated as low-rank
        err_tol (float): convergence criteria for models
        maxit (int): maximum number of iterations for models
        rho_upd (float): _description_
        rho_upd_thr (float): _description_
        graph_modes (list of int): Which mode of the tensor corresponds to the graph.

    Returns:
        independent_vars, control_vars, runtime_misc, dependent_variables, directions, model_runners
    """
    ### DATA AND MODEL RUNNERS
    independent_vars = {
        'model': {'type': 'categorical', 'choices': ['HoRPCA', 'SNN_LOGS(N1)', 'SNN_LOGS(/N1)', 'SNN_LOGS(-E)' ]},
        'spread': {'type': 'categorical', 'choices': ['isotropic', 'anisotropic']}, #anomaly_type, # 'isotropic' or 'anisotropic'}
        'radius': {'type': 'int', 'low': 0, 'high': 100},
        # 'NoA': {'type': 'int', 'low': 0, 'high': 10000},
    }
    control_vars = {
        'dim': dim,
        'ranks': ranks,
        'graph_type': graph_type, # 'Gnm', 'random_regular'
        'graph_dim': graph_dim, # (10,10) # {'graph_type': , 'n': n, 'm': m, 'directed': False} # {'graph_type': 'Gnm', 'n': n, 'm': m, 'directed': True} # {'graph_type': , 'd': d, 'n': n}
        'NoA': NoA,
        'local_dist': local_dist,
        'time_m': time_m,
        'local_m': local_m,
        'duration': duration,
        'amplitude': amplitude
    }
    runtime_misc = ['seed']
    dependent_variables = ['tol', 'diff', 'L_err', 'S_err', 'AUC-PRC', 'AUC-ROC']
    directions = ['minimize', 'minimize', 'minimize', 'minimize', 'maximize', 'maximize']
    ### HoRPCA
    horpca_control_vars = {
        'eps': eps,
        'lr_modes': lr_modes,
        'err_tol': err_tol,
        'maxit': maxit,
        'step_size_growth': rho_upd,
        'mu': rho_upd_thr,
        'verbose': 0,
    }
    horpca_independent_vars = {
        't': {'type': 'float', 'low': 0, 'high': 1},
    }
    horpca_runtime_misc = ['device']
    #### SNN_LOGS
    snn_logs_independent_vars = {
        't': {'type': 'float', 'low': 0, 'high': 1},
        'grouping': {'type': 'categorical', 'choices': ['neighbor', 'edge']},
        'weighing': {'type': 'categorical', 'choices': ['size_normalized', 'size_normalized_inv', 'uniform']},
        'r_hop': {'type': 'int', 'low': 1, 'high': 3},
    }
    snn_logs_runtime_misc = ['device']
    snn_logs_control_vars = {
        'lr_modes': lr_modes,
        'graph_modes': graph_modes,
        'max_iter': maxit,
        'err_tol': err_tol,
        'rho_update': rho_upd,
        'rho_update_thr': rho_upd_thr,
        'eps':eps,
    }
    
    model_runners = {
        'HoRPCA': HoRPCARunner(horpca_control_vars, horpca_independent_vars, horpca_runtime_misc),
        'SNN_LOGS(N1)' : SNN_LOGSRunner(snn_logs_control_vars, snn_logs_independent_vars, snn_logs_runtime_misc),
        'SNN_LOGS(/N1)': SNN_LOGSRunner(snn_logs_control_vars, snn_logs_independent_vars, snn_logs_runtime_misc),
        'SNN_LOGS(-E)' : SNN_LOGSRunner(snn_logs_control_vars, snn_logs_independent_vars, snn_logs_runtime_misc)
    }
    return independent_vars, control_vars, dependent_variables, runtime_misc, model_runners, directions


class Experiment1Runner(ExperimentTrialRunnerBase):

    def _get_data(self, variables):
        seed = variables['seed']
        # data_gen_param['lr_param'] = {'dim': [d1,d2,d3], 'ranks': [r1,r2,r3]}
        X_gt = generate_low_rank_data(variables['dim'], variables['ranks'], seed=seed)
        
        graph_type = variables['graph_type']
        if graph_type == 'grid':
            G = nx.grid_2d_graph(*variables['graph_dim'], periodic=False)
        elif graph_type == 'Gnm':
            G = nx.gnm_random_graph(variables['Gnm_n'], variables['Gnm_m'], 
                                    directed=variables['G_directed'], seed=variables['graph_seed'])
        elif graph_type == 'random_regular':
            G = nx.random_regular_graph(variables['Grr_d'], variables['Grr_n'],
                                        seed=variables['graph_seed'])
        else:
            raise ValueError("Invalid graph type.")
        
        S_gt, labels = generate_spatio_temporal_anomaly(X_gt.shape, G, 
                            variables['NoA'],
                            amplitude=variables['amplitude'],
                            local_dist=variables['local_dist'],
                            time_m=variables['time_m'], local_m=variables['local_m'],
                            duration=variables['duration'], radius=variables['radius'], 
                            anomaly_spread=variables['spread'], seed=seed)
        Y = X_gt + S_gt
        return {'Y': Y, 'X_gt': X_gt, 'S_gt': S_gt, 'G': G, 'labels': labels}
    
    def _calculate_metrics(self, data, results):
        model = results[0]
        model_eps = results[1]
        X_gt = data['X_gt']
        S_gt = data['S_gt']
        labels = data['labels']
        device = model.device
        fpr, tpr, thresholds = roc_curve(labels.ravel(),
                                        torch.abs(model.S).ravel().cpu().numpy())
        precision, recall, thresholds = precision_recall_curve(labels.ravel(),
                                            torch.abs(model.S).ravel().cpu().numpy())
        auc_prc_score = auc(recall, precision)
        auc_roc_score = auc(fpr, tpr)

        metrics = {'AUC-ROC': auc_roc_score,
                    'AUC-PRC': auc_prc_score,
                    'S_diff': torch.norm(model.S-model_eps.S).cpu().item(),
                    'L_diff': torch.norm(model.X-model_eps.X).cpu().item(),
                    'L_nuc_1': torch.linalg.matrix_norm(t2m(model.X, 1), 'nuc').cpu().item(),
                    'L_nuc_2': torch.linalg.matrix_norm(t2m(model.X, 2), 'nuc').cpu().item(),
                    'L_nuc_3': torch.linalg.matrix_norm(t2m(model.X, 3), 'nuc').cpu().item(),
                    'S_nuc_1': torch.linalg.matrix_norm(t2m(model.S, 1), 'nuc').cpu().item(),
                    'S_nuc_2': torch.linalg.matrix_norm(t2m(model.S, 2), 'nuc').cpu().item(),
                    'S_nuc_3': torch.linalg.matrix_norm(t2m(model.S, 3), 'nuc').cpu().item(),
                    'L_nuc_1_eps': torch.linalg.matrix_norm(t2m(model_eps.X, 1), 'nuc').cpu().item(),
                    'L_nuc_2_eps': torch.linalg.matrix_norm(t2m(model_eps.X, 2), 'nuc').cpu().item(),
                    'L_nuc_3_eps': torch.linalg.matrix_norm(t2m(model_eps.X, 3), 'nuc').cpu().item(),
                    'S_nuc_1_eps': torch.linalg.matrix_norm(t2m(model_eps.S, 1), 'nuc').cpu().item(),
                    'S_nuc_2_eps': torch.linalg.matrix_norm(t2m(model_eps.S, 2), 'nuc').cpu().item(),
                    'S_nuc_3_eps': torch.linalg.matrix_norm(t2m(model_eps.S, 3), 'nuc').cpu().item(),
                    'L1':  torch.sum(torch.abs(model.X)).cpu().item(),
                    'L1_eps':  torch.sum(torch.abs(model_eps.X)).cpu().item(),
                    'S1': torch.sum(torch.abs(model.S)).cpu().item(),
                    'S1_eps': torch.sum(torch.abs(model_eps.S)).cpu().item(),
                    'nonzero_S': (torch.sum(model.S != 0)/torch.prod(torch.tensor(model.S.shape))).cpu().item(),
                    'nonzero_L': (torch.sum(model.X != 0)/torch.prod(torch.tensor(model.S.shape))).cpu().item(),
                    'ranks_S1': torch.linalg.matrix_rank(t2m(model.S, 1)).cpu().item(),
                    'ranks_S2': torch.linalg.matrix_rank(t2m(model.S, 2)).cpu().item(),
                    'ranks_S3': torch.linalg.matrix_rank(t2m(model.S, 3)).cpu().item(),
                    'ranks_L1': torch.linalg.matrix_rank(t2m(model.X, 1)).cpu().item(),
                    'ranks_l2': torch.linalg.matrix_rank(t2m(model.S, 2)).cpu().item(),
                    'ranks_l3': torch.linalg.matrix_rank(t2m(model.S, 3)).cpu().item(),
                    'S_fro': torch.norm(model.S).cpu().item(),
                    'L_fro': torch.norm(model.X).cpu().item(),
                    'L_fro_eps': torch.norm(model_eps.X).cpu().item(),
                    'S_fro_eps': torch.norm(model_eps.S).cpu().item(),
                    'S_err': (torch.norm(torch.tensor(S_gt, device=device)-model.S).cpu().numpy()/np.linalg.norm(S_gt)).item(),
                    'L_err': (torch.norm(torch.tensor(X_gt, device=device)-model.X).cpu().numpy()/np.linalg.norm(X_gt)).item()
                    }
        metrics['tol'] = metrics['S_err'] + metrics['L_err']
        metrics['diff'] = metrics['S_diff'] + metrics['L_diff']
        return metrics

    def study_name_parser(self, exp_name, variables_of_interest):
        """Parse the study name and return the variables of interest"""
        study_name = f'{exp_name}'
        return study_name

class SNN_LOGSRunner(ModelRunnerBase):
    def __init__(self, model_control_vars,
                 model_independent_vars,
                 model_runtime_misc):
        super().__init__(model_control_vars, model_independent_vars, model_runtime_misc)

    
    def _run(self, data, variables):
        G = data['G']
        Y = data['Y']
        t = variables['t']
        eps = variables['eps']

        var1 = {}
        var1['lr_modes'] = variables['lr_modes']
        var1['graph_modes'] = variables['graph_modes']
        var1['grouping'] = variables['grouping']
        var1['weighing'] = variables['weighing']
        var1['r_hop'] = variables['r_hop']
        var1['device'] = variables.get('device', None)
        var1['dtype'] = variables.get('dtype', torch.float64)
        var1['verbose'] = variables.get('verbose', 0)
        model1 = SNN_LOGS(Y, G, **var1)
        model2 = SNN_LOGS(Y, G, **var1)
        var2 = {}
        var2['psis'] = [1-t]*len(variables['lr_modes'])
        var2['lda'] = t
        var2['max_iter'] = variables.get('max_iter', 150)
        var2['rho'] = variables.get('rho', 4*np.abs(Y).sum()/Y.size)
        var2['err_tol'] = variables.get('err_tol', 1e-6)
        var2['rho_update'] = variables.get('rho_update', 1)
        var2['rho_update_thr'] = variables.get('rho_update_thr', 100)
        X1, S1 = model1(**var2)
        var2['psis'] = [1-t+eps]*len(variables['lr_modes'])
        var2['lda'] = t - eps
        X2, S2 = model2(**var2)
        return model1, model2

    @property
    def model_variable_names(self):
        return ['Y', 'G', 'lr_modes', 'graph_modes', 'grouping', 'weighing',
                'r_hop', 'device', 'dtype', 'verbose'] + \
                ['psis', 'lda', 'max_iter', 'rho', 'err_tol', 'rho_update',
                 'rho_update_thr']
    
    @property
    def name(self):
        return "SNN_LOGS"


class HoRPCARunner(ModelRunnerBase):
    def __init__(self, model_control_vars,
                 model_independent_vars,
                 model_runtime_misc):
        super().__init__(model_control_vars, model_independent_vars, model_runtime_misc)


    def _run(self, data, variables):
        Y = data['Y']
        var = {}
        t = variables['t']
        eps = variables['eps']
        var['lr_modes'] = variables['lr_modes']
        var['lda1'] = t
        var['lda_nucs'] = [1-t]*len(variables['lr_modes'])
        var['rho'] = variables.get('rho', 4*np.abs(Y).sum()/Y.size)
        var['err_tol'] = variables.get('err_tol', 1e-6)
        var['maxit'] = variables.get('maxit', 300)
        var['step_size_growth'] = variables.get('step_size_growth', variables.get('rho_update', 1))
        var['mu'] = variables.get('mu', variables.get('rho_update_thr', 100))
        var['verbose'] = variables.get('verbose', 0)
        var['device'] = variables.get('device', None)
        var['dtype'] = variables.get('dtype', torch.float64)
        
        model1 = HoRPCA_Singleton(Y, **var)
        X1,S1 = model1()
        
        var['lda1'] = t - eps
        var['lda_nucs'] = [1-t+eps]*len(variables['lr_modes'])
        model2 = HoRPCA_Singleton(Y, **var)
        X2,S2 = model2()
        return model1, model2
    
    @property
    def model_variable_names(self):
        return ['Y', 'lda1', 'lda_nucs', 'rho', 'err_tol', 'maxit', 'step_size_growth', 'mu',
                'verbose', 'device', 'dtype', 'lr_modes', 'report_freq', 'metric_tracker']

    @property
    def name(self):
        return "HoRPCA"