from pprint import pprint
import os, sys
from copy import deepcopy
from collections import defaultdict

module_path = os.path.abspath(os.path.join('..','..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import wandb
from dask.distributed import Client, as_completed
import networkx as nx
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import optuna

from src.models.lr_ssd.snn__logn_gtv import SNN__LOGN_GTV
from data.nyc_taxi_dataset import NYCTaxiDataset
from src.stats.multi_linear_normal import MultiLinearNormal

# optuna.logging.set_verbosity(optuna.logging.ERROR)

PERCENTAGES = ['3%', '2%', '1%', '0.7%', '0.3%','0.14%','0.07%','0.014%']
PERCENTAGES = PERCENTAGES[::-1]


def nyc_hp_study(study_config, model_config, step=None):
    study_name = model_config['name'] + '_'+ '_'.join(study_config['metrics'])
    if study_config['storage'] is None:
        study = optuna.create_study(directions=study_config['directions'],
                                    sampler=optuna.samplers.TPESampler())
    else:
        if study_config['overwrite']:
            optuna.study.delete_study(study_name=study_name, storage=study_config['storage'])
        study = optuna.study.create_study(study_name=study_name,
                                            directions= study_config['directions'],
                                            storage= study_config['storage'],
                                            load_if_exists= True,
                                            sampler=optuna.samplers.TPESampler())

    
    objective = SNN_LOGN_GTVObjective(study_config, model_config)
    
    study.optimize(objective, n_trials=study_config['n_trials'], show_progress_bar=True)

    if len(study_config['metrics']) == 1:
        best_trial = study.best_trial
    elif len(study_config['metrics']) == 2:
        if study_config['directions'][0] == 'maximize':
            best_trial = max(study.best_trials, key=lambda t: t.values[0])
        elif study_config['directions'][0] == 'minimize':
            best_trial = min(study.best_trials, key=lambda t: t.values[0])
        best_result = (best_trial.user_attrs['metrics'] | best_trial.params)
    return best_result
    


class SNN_LOGN_GTVObjective:
    def __init__(self, study_config, model_config):
        self.data = NYCTaxiDataset()
        self.study_config = study_config
        self.model_config = model_config
        
        G = self.data.G_nyc
        if model_config['graph'] == 'temporal':
            G = self.data.Gt
        elif model_config['graph'] == 'spatio-temporal':
            Gt = self.data.Gt
            pd = model_config.get('graph_product', 'cartesian')
            if pd == 'cartesian':
                G = nx.cartesian_product(G, Gt)
            elif pd == 'strong':
                G = nx.strong_product(G, Gt)
        self.G = G

        self.Y = np.moveaxis(self.data.dropoffs, [0,1], [3,0]) # (53, 7, 24, 81)
        # np.moveaxis(Y,[0,1,2,3],[1,2,3,0]) Reverses
        self.model_select = {
            'lr_modes': self.model_config['lr_modes'],
            'graph_modes': self.model_config['graph_modes'],
            'grouping': self.model_config['grouping'],
            'weighing': self.model_config['weighing'],
            'r_hop': self.model_config['r_hop'],
            'device': self.model_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'),
            'dtype': self.model_config.get('dtype', torch.float64),
            'verbose': self.model_config.get('verbose', 0),
            'gtvr_config': self.model_config['gtvr_config'],
            'soft_constrained': self.model_config.get('soft_constrained', False),
            }
        self.model_run = {
            'max_iter': self.model_config.get('max_iter', 150),
            'rho': self.model_config.get('rho', 0.001),
            'err_tol': self.model_config.get('err_tol', 1e-6),
            'rho_update': self.model_config.get('rho_update', 1),
            'rho_update_thr': self.model_config.get('rho_update_thr', 100)
        }
        

    def __call__(self, trial):
        ts = [-np.log(trial.suggest_float(f"psi_{i}", 0, 1)) for i in range(
                        len(self.model_config['lr_modes']))
                        ] + [-np.log(trial.suggest_float("lda", 0, 1))]
        if self.model_config['soft_constrained']:
            ts += [-np.log(trial.suggest_float(f"lda_f", 0, 1))]
        
        ts = np.array(ts)
        ts = list(ts/np.sum(ts))
        if self.model_config['soft_constrained']:
            alg_params = {'psis': ts[:-2],
                          'lda': ts[-2],
                          'lda_f': ts[-1]}
        else:
            alg_params = {'psis': ts[:-1],
                          'lda': ts[-1]}
        lda_gtv_min = self.study_config.get('lda_gtv_min',1e-8)
        lda_gtv_max = self.study_config.get('lda_gtv_max',10)
        lda_gtv_log = self.study_config.get('lda_gtv_log', True)
        lda_gtvs = [trial.suggest_float(f"lda_gtv_{i}", lda_gtv_min, lda_gtv_max, log=lda_gtv_log
                                        ) for i in range(len(self.model_config['gtvr_config']))]
        alg_params['lda_gtvs'] = lda_gtvs
        trial.set_user_attr("alg_params", alg_params)
        model = SNN__LOGN_GTV(self.data['Y'], self.G, **self.model_select)
        X, S = model(ts[:-1], ts[-1], lda_gtvs, **self.model_run)
        
        mln_nyc = MultiLinearNormal((7,24,81), device=self.model_select['device'], dtype=torch.float64)
        mln_nyc.mle(model.X, verbose=0);

        metrics = self.calculate_metrics(model, mln_nyc)
        trial.set_user_attr("metrics", metrics)

        if len(self.study_config['metrics']) == 1:
            return metrics[self.study_config['metrics'][0]]
        elif len(self.study_config['metrics']) == 2:
            return metrics[self.study_config['metrics'][0]], metrics[self.study_config['metrics'][1]]


    def calculate_metrics(self, model, mln_nyc):
        metrics = model.bic2() # BIC2, NLL, obj, num_parameters
        metrics['primal_residual'] = model.r[-1].cpu().item()
        metrics['dual_residual'] = model.s[-1].cpu().item()
        metrics['S_SPARSITY'] = (torch.sum(model.S.abs()> 1e-5).cpu().item()/torch.prod(torch.tensor(model.S.shape))).cpu().item()
        
        ratios = np.array([0.014, 0.07, 0.14, 0.3, 0.7, 1, 2, 3])/100
        score = torch.abs(model.S).cpu().numpy()
        vanilla_score_detection = np.array(
            [sum(self.data.detect_topk_events(np.moveaxis(score,[0,1,2,3],[1,2,3,0]), ratio))
             for ratio in ratios]
             )
        score = (model.Y - mln_nyc.mean).abs().cpu().numpy()
        mln_score_detection = np.array(
            [sum(self.data.detect_topk_events(np.moveaxis(score,[0,1,2,3],[1,2,3,0]), ratio))
             for ratio in ratios]
             )
        for i, pcntg in enumerate(PERCENTAGES):
            metrics[pcntg] = vanilla_score_detection[i]
            metrics['MLN_'+pcntg] = mln_score_detection[i]
        metrics['detected_events'] = vanilla_score_detection
        metrics['Total'] = sum(vanilla_score_detection)
        metrics['MLN_detected_events'] = mln_score_detection
        metrics['MLN_Total'] = sum(mln_score_detection)
        return metrics
        

    




class NYCTaxiHPStudy:
    def __init__(self, config, api_key, 
                client=Client(n_workers=5),
                **kwargs):
        self.exp_config = config
        self.all_results = defaultdict(list)
        wandb.login(key=api_key, verify=True)
        self.run_kwargs = kwargs
        self.num_active_remote = 0
        self.client = client
        

    def run_study(self, lr_modes, data='dropoffs'):
        print("Experiment Configuration: ") 
        pprint(self.exp_config)
        models = self.exp_config['models']

        for key, model_variables in models.items():
            tags = self.tags + [model_variables['name']]
            

            for i, lr_mode in tqdm(enumerate(lr_modes), 
                                    total=len(lr_modes), 
                                    desc=f"Model: {model_variables['name']}"):
                
                run = wandb.init(project=self.project_name,
                                    group=self.group_name,
                                    config={
                                        'data': data_variables,
                                        'model': model_var,
                                        'independent_var': self.exp_config['independent_var'],
                                        'mode_swept': lr_mode
                                    },
                                    name=f"{model_variables['name']}_modes_{lr_mode}",
                                    tags=tags,
                                    **self.run_kwargs)
                
                wandb.define_metric('t')
                for metric in METRICS:
                    wandb.define_metric(metric, step_metric='t')
                
                obj_refs = []
                for j, t in enumerate(t_range):
                    model_var['t'] = t
                    model_var['device'] = f'cuda:{j%torch.cuda.device_count()}'
                    obj_refs.append(
                        self.client.submit(model_runner, model_variables['model'], data_variables, model_var, t)
                        )
                    self.num_active_remote += 1
                    # Ensure there are always 4 tasks running
                    if self.num_active_remote >= torch.cuda.device_count():
                        while len(obj_refs) >= torch.cuda.device_count():
                            for future in as_completed(obj_refs):
                                result, t = future.result()
                                result['t'] = t
                                wandb.log(result)
                                
                                self.all_results[model_variables['name']].append(result)
                                obj_refs.remove(future)
                                self.num_active_remote -= 1
                    
                # Process remaining tasks
                for future in as_completed(obj_refs):
                    result, t = future.result()
                    result['t'] = t
                    wandb.log(result)
                    result['lr_mode'] = lr_mode
                    self.all_results[model_variables['name']].append(result)
                run.finish()
        return self.all_results