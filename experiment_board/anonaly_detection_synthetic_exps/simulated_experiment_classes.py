from pprint import pprint
import os, sys
from pathlib import Path

from copy import deepcopy
from collections import defaultdict

import wandb
# import ray
from dask.distributed import Client, as_completed
import networkx as nx
import torch
import numpy as np
import numpy.linalg as npla
import pandas as pd
# from sklearn.metrics import precision_recall_curve, auc, roc_curve
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, precision_recall_curve, auc, roc_curve
from tqdm import tqdm
import optuna

from src.multilinear_ops.matricize import matricize
from src.models.horpca.horpca_torch import HoRPCA_Singleton
from src.models.lr_ssd.snn__logn_gtv import SNN__LOGN_GTV

from src.synthetic_data.generate_lr_data import generate_low_rank_data
from src.synthetic_data.generate_anomaly import generate_spatio_temporal_anomaly
from src.multilinear_ops.matricize import matricize
from src.multilinear_ops.tensorize import tensorize

optuna.logging.set_verbosity(optuna.logging.ERROR)

class Experiment:
    def __init__(self, api_key, exp_config, n_workers=4,
                 ):
        self.exp_config = exp_config
        self.all_results = []
        wandb.login(key=api_key, verify=True)
        self.num_active_remote = 0
        self.client = Client(n_workers=n_workers)
        self.result_csv_path = self.exp_config['result_csv_path']

    def run_experiment(self, seed, n_trials, message=None):
        print("Experiment Configuration:") 
        pprint(self.exp_config)

        models = self.exp_config['models']
        independent_var = self.exp_config['independent_var']
        dependent_var = self.exp_config['dependent_var']
        study_config = self.exp_config['study_config']

        for key, model_variables in models.items():
            print(f"Running model: {key}")
            print("--"*20)
            tags = deepcopy(self.exp_config['tags'])
            tags = tags + [model_variables['name'], self.exp_config['group_name'], self.exp_config['config_name']]
            tags = tags + [f'study_metric{ii}_{metric}' for ii, metric in enumerate(study_config['metrics'])]
            tags = list(set(tags))

            notes = str(tags) + '\n---------\n' + self.exp_config['notes'] + '\n------ Message ------\n' + message
            for i in tqdm(range(n_trials),
                          desc=f"Running {model_variables['name']}", total=n_trials):
                trial_results = []
                data_var = deepcopy(self.exp_config['data'])
                data_var['seed'] = seed + i
                run_name = f"{self.exp_config['group_name']}_{self.exp_config['config_name']}_{model_variables['name']}_{data_var['seed']}"
                run = wandb.init(project=self.exp_config['project_name'],
                                    group=self.exp_config['group_name'],
                                    name=run_name,
                                    config={
                                        'data': data_var,
                                        'model': model_variables,
                                        'independent_var': independent_var,
                                        'study_config': study_config,
                                    },
                                    tags=tags,
                                    notes=notes)
                
                independent_var_name = independent_var['keys'][-1]
                independent_var_range = independent_var['range']
                wandb.define_metric(independent_var_name)
                for dep_var in dependent_var:
                    wandb.define_metric(dep_var, step_metric=independent_var_name)

                obj_refs = []
                for j, ind_var_value in enumerate(independent_var_range):
                    data_var = set_nested_value(data_var, independent_var['keys'], ind_var_value)

                    model_variables['device'] = f'cuda:{j%torch.cuda.device_count()}'
                    
                    obj_refs.append(
                        self.client.submit(hp_study, run_name, study_config, model_variables, data_var, ind_var_value)
                        )
                    self.num_active_remote += 1
                    # Ensure there are always 4 tasks running
                    if self.num_active_remote >= torch.cuda.device_count():
                        while len(obj_refs) >= torch.cuda.device_count():
                            for future in as_completed(obj_refs):
                                result, step = future.result()
                                result[independent_var_name] = step
                                wandb.log(result)
                                result['seed'] = data_var['seed']
                                result['model'] = model_variables['name']
                                trial_results.append(result)

                                df = pd.DataFrame([result])
                                RESULT_DIR = Path(self.result_csv_path)
                                if not RESULT_DIR.exists():
                                    RESULT_DIR.mkdir(parents=True, exist_ok=True)
                                model_result_path = RESULT_DIR / f"{self.exp_config['config_name']}_{model_variables['name']}.csv"
                                if os.path.exists(model_result_path):
                                    df.to_csv(model_result_path, mode='a', header=False, index=False)
                                else:
                                    df.to_csv(model_result_path, mode='w', header=True, index=False)

                                # self.all_results[model_variables['name']].append(result)
                                obj_refs.remove(future)
                                self.num_active_remote -= 1
                    
                # Process remaining tasks
                for future in as_completed(obj_refs):
                    result, step = future.result()
                    result[independent_var_name] = step
                    wandb.log(result)
                    result['seed'] = data_var['seed']
                    result['model'] = model_variables['name']
                    trial_results.append(result)
                    df = pd.DataFrame(result)
                    RESULT_DIR = Path(self.result_csv_path)
                    if not RESULT_DIR.exists():
                        RESULT_DIR.mkdir(parents=True, exist_ok=True)
                    model_result_path = RESULT_DIR / f"{self.exp_config['config_name']}_{model_variables['name']}.csv"
                    if os.path.exists(model_result_path):
                        df.to_csv(model_result_path, mode='a', header=False, index=False)
                    else:
                        df.to_csv(model_result_path, mode='w', header=True, index=False)
                run.finish()

                # df = pd.DataFrame(trial_results)
                # # Append the results to a CSV file
                # if os.path.exists(self.result_csv_path):
                #     df.to_csv(self.result_csv_path, mode='a', header=False, index=False)
                # else:
                #     df.to_csv(self.result_csv_path, mode='w', header=True, index=False)

                self.all_results.append(trial_results)
        

def study_callback(study, frozen_trial):
    """Logging callback that reports when the Pareto front is updated."""
    current_pareto_front = study.best_trials  # Get the current Pareto front
    updated = False
    # Check if the current trial is part of the updated Pareto front
    for trial in current_pareto_front:
        if trial.number == frozen_trial.number:
            updated = True
            break
    if updated:
        print(f"----- Pareto front updated with trial {frozen_trial.number} -----")
        print(f"Values: {frozen_trial.values} \t Params: {frozen_trial.params}")


def hp_study(run_name, study_config, model_config, data_variables, step=None):
    study_name = run_name + f'_{step}'
    sampler_config = study_config.get('sampler_config', None)
    if sampler_config is not None:
        sampler_name = sampler_config['type']
        if sampler_name == 'TPESampler':
            sampler = optuna.samplers.TPESampler(**sampler_config.get('params', {}))
        elif sampler_name == 'CmaEsSampler':
            sampler = optuna.samplers.CmaEsSampler(**sampler_config.get('params', {}))
        elif sampler_name == 'GPSampler':
            sampler = optuna.samplers.GPSampler(**sampler_config.get('params', {}))
        elif sampler_name == 'RandomSampler':
            sampler = optuna.samplers.RandomSampler(**sampler_config.get('params', {}))
        else:
            raise ValueError(f"Unsupported sampler: {sampler_name}")
    else:
        sampler = None
            
    if study_config['storage'] is None:
        if len(study_config['directions'])>1:
            study = optuna.create_study(directions=study_config['directions'],
                                        sampler=sampler,
                                        study_name=study_name,
                                        load_if_exists=True)
        else:
            study = optuna.create_study(direction=study_config['directions'][0],
                                        sampler=sampler,
                                        study_name=study_name,
                                        load_if_exists=True)
    else:
        if len(study_config['directions'])>1:
            if study_config['overwrite']:
                optuna.study.delete_study(study_name=study_name, storage=study_config['storage'])
            study = optuna.study.create_study(study_name=study_name,
                                                sampler=sampler,
                                                directions= study_config['directions'],
                                                storage= study_config['storage'],
                                                load_if_exists= True)
        else:
            if study_config['overwrite']:
                optuna.study.delete_study(study_name=study_name, storage=study_config['storage'])
            study = optuna.study.create_study(study_name=study_name,
                                                sampler=optuna.samplers.GPSampler(),
                                                direction= study_config['directions'][0],
                                                storage= study_config['storage'],
                                                load_if_exists= True)

    if model_config['model'] == 'HoRPCA':
        objective = HoRPCAObjective(data_variables, study_config, model_config)
    elif model_config['model'] == 'SNN_LOGN_GTV':
        objective = SNN_LOGN_GTVObjective(data_variables, study_config, model_config)
    else:
        raise ValueError('Invalid model type')
    
    study.optimize(objective, n_trials=study_config['n_trials'], show_progress_bar=True)
    
    importances = {}
    if study_config['n_trials'] > 1:
        parameter_importances = optuna.importance.get_param_importances(study, normalize=True)
        for param, importance in parameter_importances.items():
            importances[param+ '_importance'] = importance
    
    if len(study_config['directions'])>1:
        if study_config['directions'][0] == 'maximize':
            best_trial = max(study.best_trials, key=lambda t: t.values[0])
        elif study_config['directions'][0] == 'minimize':
            best_trial = min(study.best_trials, key=lambda t: t.values[0])
    else:
        best_trial = study.best_trial
    study.set_user_attr('importances', importances)
    results = (best_trial.user_attrs['metrics'] | best_trial.params | importances)
    return results, step
    

class HoRPCAObjective:
    def __init__(self, data_variables, study_config, model_config):
        self.data = get_data(data_variables)
        self.study_config = study_config
        self.model_config = model_config
    
    def __call__(self, trial):
        # ts = np.array([-np.log(trial.suggest_float(f"psi_{i}", 0, 1)) for i in range(
        #                 len(self.model_config['lr_modes']))]+
        #                 [-np.log(trial.suggest_float("lda1", 0, 1))])
        # ts = list(ts/np.sum(ts))
        # alg_params = {'lda_nucs': ts[:-1],
        #                 'lda1': ts[-1]}
        trial.set_user_attr("alg_params", alg_params)
        model = HoRPCA_Singleton(self.data['Y'], **self.model_config, **alg_params)
        X, S = model()
        metrics = calculate_metrics({'model': model}, self.data)
        trial.set_user_attr("metrics", metrics)
        if len(self.study_config['metrics'])==1:
            return metrics[self.study_config['metrics'][0]]
        elif len(self.study_config['metrics'])==2:
            return metrics[self.study_config['metrics'][0]], metrics[self.study_config['metrics'][1]]

class SNN_LOGN_GTVObjective:
    def __init__(self, data_variables, study_config, model_config):
        repeated = data_variables.get('repeated', False)
        if repeated is False:
            self.data = get_data(data_variables)
        elif isinstance(repeated, int):
            self.data = []
            for i in range(repeated):
                dv = deepcopy(data_variables)
                dv['seed'] = data_variables['seed'] + i* 1000
                self.data.append(get_data(dv))
        else:
            raise ValueError('Invalid value for repeated. It should be either False or an integer.')
        self.study_config = study_config
        self.model_config = model_config

        if isinstance(self.data, list):
            G = self.data[0]['G']
            if model_config['graph'] == 'temporal':
                G = self.data[0]['Gt']
            elif model_config['graph'] == 'spatio-temporal':
                Gt = self.data[0]['Gt']
                pd = model_config.get('graph_product', 'cartesian')
                if pd == 'cartesian':
                    G = nx.cartesian_product(G, Gt)
                elif pd == 'strong':
                    G = nx.strong_product(G, Gt)
        else:
            G = self.data['G']
            if model_config['graph'] == 'temporal':
                G = self.data['Gt']
            elif model_config['graph'] == 'spatio-temporal':
                Gt = self.data['Gt']
                pd = model_config.get('graph_product', 'cartesian')
                if pd == 'cartesian':
                    G = nx.cartesian_product(G, Gt)
                elif pd == 'strong':
                    G = nx.strong_product(G, Gt)
        self.G = G

        self.model_select = {
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
            'max_iter': self.model_config.get('max_iter', 150),
            'rho': self.model_config.get('rho', 0.03),#4*np.abs(self.data['Y']).sum()/self.data['Y'].size),
            'err_tol': self.model_config.get('err_tol', 1e-6),
            'rho_update': self.model_config.get('rho_update', 1),
            'rho_update_thr': self.model_config.get('rho_update_thr', 100)
        }

    def __call__(self, trial):
        # ts = np.array([-np.log(trial.suggest_float(f"psi_{i}", 0, 1)) for i in range(
        #                 len(self.model_config['lr_modes']))
        #                 ] + [-np.log(trial.suggest_float("lda", 0, 1))])
        # ts = list(ts/np.sum(ts))
        hyperparameters = self.model_config.get('hyperparameters', None)
        if hyperparameters is not None:
            thetas = np.array(hyperparameters['thetas'])
            thetas = thetas.reshape((1, len(thetas)))
            sin_thetas = np.concatenate([np.ones((1,1)), np.sin(thetas)], axis=1)
            cos_thetas = np.concatenate([np.cos(thetas), np.ones((1,1))], axis=1)
            ts = (np.cumprod(sin_thetas, axis=1)*cos_thetas)**2
            lda_gtvs = hyperparameters.get('lda_gtvs', None)
            alg_params = {}
        else:
            if self.model_config.get('soft_constrained', False):
                ts = list(sample_from_N_simplex(trial, len(self.model_config['lr_modes'])+2).ravel())
                alg_params = {'lda_f': ts[0],
                            'lda': ts[1],
                                'psis': ts[2:]}
            else:
                ts = list(sample_from_N_simplex(trial, len(self.model_config['lr_modes'])+1).ravel())
                alg_params = {'psis': ts[1:],
                            'lda': ts[0]}
        lda_gtv_min = self.study_config.get('lda_gtv_min',1e-8)
        lda_gtv_max = self.study_config.get('lda_gtv_max',10)
        lda_gtv_log = self.study_config.get('lda_gtv_log', True)
        lda_gtvs = [trial.suggest_float(f"lda_gtv_{i}", lda_gtv_min, lda_gtv_max, log=lda_gtv_log
                                        ) for i in range(len(self.model_config['gtvr_config']))]
        if self.study_config.get('search_rho', False):
            rho_min = self.study_config.get('rho_min', 1e-5)
            rho_max = self.study_config.get('rho_max', 1e5)
            rho_log = self.study_config.get('rho_log', True)
            alg_params['rho'] = trial.suggest_float("rho", rho_min, rho_max, log=rho_log)
            self.model_run['rho'] = alg_params['rho']
        
        alg_params['lda_gtvs'] = lda_gtvs
        trial.set_user_attr("alg_params", alg_params)
        if isinstance(self.data, list):
            metrics = []
            for i in range(len(self.data)):
                model = SNN__LOGN_GTV(self.data[i]['Y'], self.G, **self.model_select)

                if self.model_config.get('soft_constrained', False):
                    X, S = model(ts[2:], ts[1], lda_gtvs, lda_f=ts[0], **self.model_run)
                else:
                    X, S = model(ts[1:], ts[0], lda_gtvs, **self.model_run)
                metrics.append(calculate_metrics({'model': model}, self.data[i]))
            # print(metrics)
            mean_metrics = {k: np.mean([m[k] for m in metrics]) for k in metrics[0].keys()}
            min_metrics = {'min_'+ k: np.min([m[k] for m in metrics]) for k in metrics[0].keys()}
            max_metrics = {'max_'+ k: np.max([m[k] for m in metrics]) for k in metrics[0].keys()}
            metrics = {**mean_metrics, **min_metrics, **max_metrics}
        else:
            model = SNN__LOGN_GTV(self.data['Y'], self.G, **self.model_select)

            if self.model_config.get('soft_constrained', False):
                X, S = model(ts[2:], ts[1], lda_gtvs, lda_f=ts[0], **self.model_run)
            else:
                X, S = model(ts[1:], ts[0], lda_gtvs, **self.model_run)
            # model.move_metrics_to_cpu()
            metrics = calculate_metrics({'model': model}, self.data)

        trial.set_user_attr("metrics", metrics)
        if len(self.study_config['metrics'])==1:
            return metrics[self.study_config['metrics'][0]]
        elif len(self.study_config['metrics'])==2:
            return metrics[self.study_config['metrics'][0]], metrics[self.study_config['metrics'][1]]
    
    def run_with_hyperparameters(self, hyperparameters, seed=None):
        thetas = np.array(hyperparameters['thetas'])
        thetas = thetas.reshape((1, len(thetas)))
        sin_thetas = np.concatenate([np.ones((1,1)), np.sin(thetas)], axis=1)
        cos_thetas = np.concatenate([np.cos(thetas), np.ones((1,1))], axis=1)
        ts = ((np.cumprod(sin_thetas, axis=1)*cos_thetas)**2).ravel()
        lda_gtvs = hyperparameters.get('lda_gtvs', None)

        model = SNN__LOGN_GTV(self.data['Y'], self.G, **self.model_select)

        if self.model_config.get('soft_constrained', False):
            X, S = model(ts[2:], ts[1], lda_gtvs, lda_f=ts[0], **self.model_run)
        else:
            X, S = model(ts[1:], ts[0], lda_gtvs, **self.model_run)
        
        # model.move_metrics_to_cpu()
        
        metrics = calculate_metrics({'model': model}, self.data)
        # pprint(metrics)
        if seed is not None:
            metrics['seed'] = seed
        return metrics


def get_data(data_variables):
    seed = data_variables['seed']
    lr_variables = data_variables['lowrank']
    graph_variables = data_variables['graph']
    an_variables = data_variables['anomaly']
    noise = data_variables.get('noise', None)
    train_test_split = an_variables.get('train_test_split', None)

    X_gt = generate_low_rank_data(**lr_variables, seed=seed)
    std = lr_variables.get('std', None)
    if std is not None:
        X_gt = std*X_gt/ (X_gt.std())

    gtype = graph_variables['type']
    if gtype == 'grid':
        G = nx.grid_2d_graph(graph_variables['n'], graph_variables['m'], 
                             periodic=graph_variables.get('periodic', False))
    elif gtype == 'Gnm':
        G = nx.gnm_random_graph(graph_variables['n'], graph_variables['m'],
                                seed=graph_variables['seed'],
                                directed=graph_variables['directed'])
    elif gtype == 'random_regular':
        G = nx.random_regular_graph(graph_variables['d'], graph_variables['n'],
                                    seed=graph_variables['seed'])
    else:
        raise ValueError('Invalid graph type')
    
    Gt = nx.grid_graph((X_gt.shape[an_variables['time_m']-1], ), periodic=False)
    

    if train_test_split is None:
        S_gt, labels = generate_spatio_temporal_anomaly(X_gt.shape, G, 
                                                    **an_variables,
                                                    seed=seed)
    else:
        split_mode = train_test_split['split_mode']
        train_ratio = train_test_split['train_ratio']
        num_anomalies = an_variables['num_anomalies']
        num_train_anomalies = int(num_anomalies*train_ratio)
        train_shape, test_shape = list(X_gt.shape), list(X_gt.shape)
        d = int(X_gt.shape[split_mode-1]*train_ratio)
        if d == 0:
            raise ValueError('Train ratio is too small. Please increase it.')
        train_shape[split_mode-1] = d
        test_shape[split_mode-1] = int(X_gt.shape[split_mode-1] - d)
        an_variables_train = deepcopy(an_variables)
        an_variables_test = deepcopy(an_variables)
        an_variables_train.pop('train_test_split');
        an_variables_test.pop('train_test_split');
        an_variables_train['num_anomalies'] = num_train_anomalies
        an_variables_test['num_anomalies'] = num_anomalies - num_train_anomalies
        S_train, labels_train = generate_spatio_temporal_anomaly(tuple(train_shape), G,
                                                    **an_variables_train,
                                                    seed=seed)
        S_test, labels_test = generate_spatio_temporal_anomaly(tuple(test_shape), G,
                                                    **an_variables_test,
                                                    seed=seed)
        S_gt = np.concatenate((S_train, S_test), axis=split_mode-1)
        labels = np.concatenate((labels_train, labels_test), axis=split_mode-1)
        train_mask = np.concatenate((np.ones(train_shape, dtype=bool),
                                    np.zeros(test_shape, dtype=bool)),
                                    axis=split_mode-1)

    if noise is not None:
        power_X_db = 10*np.log(np.linalg.norm(X_gt))
        power_N_db = power_X_db - noise['SNR']
        power_N = 10**(power_N_db/10)
        N = np.random.randn(*X_gt.shape)
        N = N/np.linalg.norm(N)*np.sqrt(power_N)
        Y = X_gt + S_gt + N
    else:
        Y = X_gt + S_gt
    if train_test_split is None:
        return {'Y': Y, 'X_gt': X_gt, 'S_gt': S_gt, 'G':G, 'labels': labels, 'Gt': Gt}
    else:
        return {'Y': Y, 'X_gt': X_gt, 'S_gt': S_gt, 'G':G, 'labels': labels, 'Gt': Gt,
                'S_gt_train': S_train, 'labels_train': labels_train, 'train_mask': train_mask,
                'S_gt_test': S_test, 'labels_test': labels_test}

def calculate_metrics(results, data):
    model = results['model']
    X_gt = data['X_gt']
    S_gt = data['S_gt']
    labels = data['labels']
    device = model.device
    S = model.S.cpu().numpy()
    X = model.X.cpu().numpy()
    if 'S_gt_train' in data:
        S_gt_train = data['S_gt_train']
        labels_train = data['labels_train']
        train_mask = data['train_mask']
        S_gt_test = data['S_gt_train']
        labels_test = data['labels_test']
        model_train_S = S[train_mask]
        model_test_S = S[~train_mask]
        model_train_X = X[train_mask]
        model_test_X = X[~train_mask]

    
    fpr, tpr, thresholds = roc_curve(labels.ravel(),
                                    torch.abs(model.S).ravel().cpu().numpy())
    precision, recall, thresholds = precision_recall_curve(labels.ravel(),
                                        torch.abs(model.S).ravel().cpu().numpy())
    auc_prc_score = auc(recall, precision)
    auc_roc_score = auc(fpr, tpr)
    # bic, k = model.bayesian_information_criterion() # old
    metrics = model.num_parameters()
    metrics['AUC-ROC'] = auc_roc_score
    metrics['AUC-PRC'] = auc_prc_score
    metrics['S_err'] =  torch.norm(torch.tensor(S_gt, device=device)-model.S).cpu().numpy().item()#/torch.norm(torch.tensor(S_gt, device=device)))).cpu().numpy().item()
    metrics['L_err'] =  torch.norm(torch.tensor(X_gt, device=device)-model.X).cpu().numpy().item()#/torch.norm(torch.tensor(X_gt, device=device)))).cpu().numpy().item()
    metrics['error'] = np.sqrt((metrics['S_err']**2 + metrics['L_err']**2))
    metrics['S_rel_err'] = metrics['S_err']/np.linalg.norm(S_gt)
    metrics['L_rel_err'] = metrics['L_err']/np.linalg.norm(X_gt)
    metrics['SL_rel_err'] = metrics['L_rel_err'] + metrics['S_rel_err']
    metrics['rel_err'] = metrics['error']/np.sqrt(np.linalg.norm(S_gt)**2 + np.linalg.norm(X_gt)**2)
    metrics['primal_residual'] = model.r[-1]
    metrics['dual_residual'] = model.s[-1]

    try:
        V_sum = model.V.sum(dim=0, keepdim=False).coalesce().to_dense()
        non_zeros = (tensorize(V_sum, model.Y.shape, model.graph_modes)!=0).cpu().numpy()
        # non_zero_abs_S = np.abs(S[non_zeros])
        if non_zeros.sum() == 0:
            t_labels = np.zeros_like(S, dtype=bool)
        else:
            non_zero_S = S[non_zeros]
            var_brac = np.quantile(np.abs(non_zero_S), 0.1)
            var = np.var(non_zero_S[np.abs(non_zero_S) < var_brac].ravel())
            t_labels = np.abs(S) > np.sqrt(var_brac)
        # Q1 = np.quantile(non_zero_abs_S, 0.25)
        # Q3 = np.quantile(non_zero_abs_S, 0.75)
        # IQR = Q3 - Q1
        # t_labels = np.abs(S) > (Q3 + 1.75 * IQR)

        # median_S = np.median(S.ravel())
        # Q1 = np.quantile(S.ravel(), 0.25)
        # Q3 = np.quantile(S.ravel(), 0.75)
        # IQR = Q3 - Q1
        # t_labels = (S < Q1 - 2.5 * IQR) | (S > Q3 + 2.5 * IQR)
        metrics['F1'] = f1_score(labels.ravel(), t_labels.ravel(), zero_division=0)
        metrics['Precision'] = precision_score(labels.ravel(), t_labels.ravel(), zero_division=0)
        metrics['Recall'] = recall_score(labels.ravel(), t_labels.ravel(), zero_division=0)
        metrics['coverage_ratio'] = np.logical_and(labels.ravel(), t_labels.ravel()).sum()/labels.sum()
    except AttributeError:
        pass


    if 'S_gt_train' in data:
        fpr_train, tpr_train, thresholds = roc_curve(labels_train.ravel(),
                                                    np.abs(model_train_S).ravel())
        precision_train, recall_train, thresholds = precision_recall_curve(labels_train.ravel(),
                                                                            np.abs(model_train_S).ravel())
        auc_prc_score_train = auc(recall_train, precision_train)
        auc_roc_score_train = auc(fpr_train, tpr_train)
        fpr_test, tpr_test, thresholds = roc_curve(labels_test.ravel(),
                                                    np.abs(model_test_S).ravel())
        precision_test, recall_test, thresholds = precision_recall_curve(labels_test.ravel(),
                                                                            np.abs(model_test_S).ravel())
        auc_prc_score_test = auc(recall_test, precision_test)
        auc_roc_score_test = auc(fpr_test, tpr_test)
        metrics['train-AUC-ROC'] = auc_roc_score_train
        metrics['train-AUC-PRC'] = auc_prc_score_train
        metrics['test-AUC-ROC'] = auc_roc_score_test
        metrics['test-AUC-PRC'] = auc_prc_score_test
        metrics['train-S_err'] = npla.norm(S_gt[train_mask]-model_train_S)/npla.norm(S_gt[train_mask])
        metrics['train-L_err'] = npla.norm(X_gt[train_mask]-model_train_X)/npla.norm(X_gt[train_mask])
        metrics['test-S_err'] = npla.norm(S_gt[~train_mask]-model_test_S)/npla.norm(S_gt[~train_mask])
        metrics['test-L_err'] = npla.norm(X_gt[~train_mask]-model_test_X)/npla.norm(X_gt[~train_mask])
    return metrics

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

def sample_from_N_simplex(trial, N):
    """Sample a point from the N-dimensional unit simplex.
    
    Favors the points that are close to 1,0,0,...,0
    Args:
        trial: The Optuna trial object.
        N: The dimension of the simplex.
    Returns:
        A point sampled from the N-dimensional unit simplex.
    """

    thetas = np.array([trial.suggest_float(f"theta_{i}", 0, np.pi/2) for i in range(1, N)]).reshape((1, N-1))
    sin_thetas = np.concatenate([np.ones((1,1)), np.sin(thetas)], axis=1)
    cos_thetas = np.concatenate([np.cos(thetas), np.ones((1,1))], axis=1)
    x = (np.cumprod(sin_thetas, axis=1)*cos_thetas)**2
    return x
