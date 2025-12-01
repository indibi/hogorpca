from pprint import pprint
# from contextlib import nullcontext
from functools import partial

import networkx as nx
import mlflow
import optuna
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.metrics import precision_recall_curve, auc, roc_curve
from dask.distributed import Client, wait
import matplotlib.pyplot as plt
from tqdm import tqdm

# from src.proximal_ops.prox_overlapping_grouped_l21 import group_indicator_matrix
from src.models.horpca.horpca_torch import HoRPCA_Singleton
from src.models.lr_ssd.snn_logs import SNN_LOGS
# from src.models.lr_ssd.lr_logs_st_tf import LR_LOGS_ST_TF
from src.synthetic_data.generate_lr_data import generate_low_rank_data
from src.synthetic_data.generate_anomaly import generate_spatio_temporal_anomaly

import warnings
warnings.filterwarnings("ignore")
# torch.use_deterministic_algorithms(True)


class IdentifiabilityStudy:
    def __init__(self, experiment_name,
                 experiment_description,
                tags=None):
        self.experiment_name = experiment_name
        self.experiment_description = experiment_description
        self.tags = tags
        self.experiment_id = get_or_create_experiment(experiment_name, tags)
        
        self.data_control_vars = None
        self.model_control_vars = None
        self.results = {}

        
    def experiment_1(self, repeat, resolution,
                    data_control_vars, mlflow_uri,
                    model_control_vars, 
                    devices,
                    eps=None, client=None, writer=None,
                    seed=0):
        if client is None:
            client = Client()
        if writer is None:
            writer = SummaryWriter('results/'+self.experiment_name)
        self.data_control_vars = data_control_vars
        self.model_control_vars = model_control_vars
        if eps is None:
            1/(resolution*40)
        self.t_range = np.linspace(eps, 1, resolution, endpoint=True)
        
        gpu_count = len(devices)

        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment(experiment_id=self.experiment_id)
        
        with mlflow.start_run(run_name=self.experiment_name, nested=True):
            mlflow.log_params(data_control_vars)
            mlflow.log_params(model_control_vars)
            mlflow.log_params({'repeat': repeat, 'resolution': resolution})
            mlflow.log_params({'seed': seed})

            
            for model in model_control_vars.keys():
                model_control_var = model_control_vars[model]
                model_param = model_control_var['model_param']
                model_select = model_control_var['model_select']
                
                with mlflow.start_run(run_name=model, nested=True):
                    mlflow.set_tags({'model': model})
                    self.results[model] = []
                    for i, t in tqdm(enumerate(self.t_range),desc=f"Model {model}"):
                        mparam = []
                        model_param['psis'] = [(1-t)]*len(model_select['lr_modes'])
                        model_param['lda_nucs'] = [(1-t)]*len(model_select['lr_modes'])
                        model_param['lda'] = 1-t
                        model_param['lda1'] = 1-t
                        mparam = [model_param.copy(), model_param.copy()]
                        mparam[1]['lda'] = 1 - t + eps
                        mparam[1]['lda1'] = 1 - t + eps
                        mparam[1]['psis'] = [(1-t+eps)]*len(model_select['lr_modes'])
                        mparam[1]['lda_nucs'] = [(1-t+eps)]*len(model_select['lr_modes'])
                        # mlflow.log_params({'t': t})

                        futures = []
                        n_active_tasks = 0
                        for r in range(repeat):
                            if n_active_tasks == gpu_count:
                                wait(futures)
                                n_active_tasks = 0
                            device = devices[n_active_tasks]
                            sd = seed + r
                            futures.append(client.submit(run_model,
                                                model_select,
                                                data_control_vars,
                                                mparam,
                                                sd,
                                                device)
                            )
                            n_active_tasks += 1
                        
                        results = client.gather(futures)

                        mean_metrics = {'mean_'+k: np.mean([r[k] for r in results]) for k in results[0].keys()}
                        std_metrics = {'var_'+k: np.std([r[k] for r in results]) for k in results[0].keys()}
                        metrics = {**mean_metrics, **std_metrics}

                        # mlflow.log_metrics(metrics, step=i)
                        writer.add_scalars(model, mean_metrics, i)
                        self.results[model].append(metrics)
                    
        return self.results

    def plot_exp1_results(self):
        for model in self.results.keys():
            results = self.results[model]
            t_range = self.t_range
            fig, ax = plt.subplots(2,3, figsize=(15,10))
            for i, m in enumerate(results[0].keys()):
                if 'mean' in m:
                    ax[i//3, i%3].errorbar(t_range, [r[m] for r in results], yerr=[r['var_'+m[5:]] for r in results])
                    ax[i//3, i%3].set_title(m[5:])
            fig.suptitle(model)
            plt.show()



def run_model(model_select, data_gen_param, model_param, seed, device):
    
    Y, X_gt, S_gt, labels, G = gen_synthetic_lr_structured_sparse_data(data_gen_param, seed)

    if model_select['name'] == 'HoRPCA':
        # model_select = {'lr_modes':[1,2,3],
        # 'verbose':0, 'mu':100, 'step_size_growth':1,
        # 'maxit':200, 'err_tol':1e-6}
        # model_param = {'lda_nucs': [0.1, 0.1, 0.1], 'lda1': 0.1}
        model_select['rho'] = model_select.get('rho',4*np.abs(Y).sum()/Y.size)
        model1 = HoRPCA_Singleton(Y, **model_select, **model_param[0], device=device)
        X1, S1 = model1()
        model2 = HoRPCA_Singleton(Y, **model_select, **model_param[1], device=device)
        X2, S2 = model2()
    elif model_select['name'] == 'SNN_LOGS':
        # model_select = {'lr_modes':[1,2,3], 'graph_modes':[1],
        # 'grouping':'neighbor', 'weighing':'size_normalized',
        # 'verbose':0,'r_hop':1}
        # model_param = {'psis': [0.1, 0.1, 0.1], 'lda': 0.1, 
        # 'rho_update':1, 'rho_update_thr':100, 'maxit':200, 'err_tol':1e-6
        mparam1 = model_param[0]
        mparam1['rho'] = mparam1.get('rho', 4*np.abs(Y).sum()/Y.size)
        model1 = SNN_LOGS(Y, G, model_select['lr_modes'], model_select['graph_modes'],
                            grouping=model_select['grouping'],
                            weighing=model_select['weighing'],
                            verbose=0, r_hop=model_select['r_hop'],
                            device=device)
        X1,S1 = model1(mparam1['psis'], mparam1['lda'], 
                        rho_update=mparam1['rho_update'], 
                        rho_update_thr=mparam1['rho_update_thr'],
                        maxit=mparam1['maxit'], 
                        err_tol=mparam1['err_tol'])
        mparam2 = model_param[1]
        mparam2['rho'] = mparam2.get('rho', 4*np.abs(Y).sum()/Y.size)
        model2 = SNN_LOGS(Y, G, model_select['lr_modes'], model_select['graph_modes'],
                            grouping=model_select['grouping'],
                            weighing=model_select['weighing'],
                            verbose=0, r_hop=model_select['r_hop'],
                            device=device)
        X2,S2 = model2(mparam2['psis'], mparam2['lda'], 
                       rho_update=mparam1['rho_update'], 
                        rho_update_thr=mparam1['rho_update_thr'],
                        maxit=mparam1['maxit'], 
                        err_tol=mparam1['err_tol'])
    else:
        raise ValueError("Invalid model name.")
    return calculate_metrics(model1, model2, X_gt, S_gt, labels)


def gen_synthetic_lr_structured_sparse_data(data_gen_param, seed):
    # data_gen_param['lr_param'] = {'dim': [d1,d2,d3], 'ranks': [r1,r2,r3]}
    X_gt = generate_low_rank_data(**data_gen_param['lr_param'], seed=seed)
    graph_setting = data_gen_param['graph_settings']
    # graph_setting = {'graph_type': 'grid', 'dim': (d1,d2)}
    #               = {'graph_type': 'Gnm', 'n': n, 'm': m, 'directed': False}
    #               = {'graph_type': 'Gnm', 'n': n, 'm': m, 'directed': True}
    #               = {'graph_type': 'random_regular', 'd': d, 'n': n}
    if graph_setting['graph_type'] == 'grid':
        G = nx.grid_2d_graph(*graph_setting['dim'], periodic=False)
    elif graph_setting['graph_type'] == 'Gnm':
        G = nx.gnm_random_graph(graph_setting['n'], graph_setting['m'], 
                                directed=graph_setting['directed'], seed=graph_setting['seed'])
    elif graph_setting['graph_type'] == 'random_regular':
        G = nx.random_regular_graph(graph_setting['d'], graph_setting['n'],
                                    seed=graph_setting['seed'])
    else:
        raise ValueError("Invalid graph type.")
    
    anomaly_param = data_gen_param['anomaly_param']
    # anomaly_param = {'NoA': NoA, 'local_dist': local_dist, 
    #               'time_m': time_m, 'local_m': local_m, 
    #               'duration': duration, 'radius': radius, 
    #               'spread': 'isotropic'}
    S_gt, labels = generate_spatio_temporal_anomaly(X_gt.shape, G, 
                        anomaly_param['NoA'],
                        amplitude=anomaly_param['amplitude'],
                        local_dist=anomaly_param['local_dist'],
                        time_m=anomaly_param['time_m'], local_m=anomaly_param['local_m'],
                        duration=anomaly_param['duration'], radius=anomaly_param['radius'], 
                        anomaly_spread=anomaly_param['spread'] , seed=seed)
    Y = X_gt + S_gt
    return Y, X_gt, S_gt, labels, G


def get_or_create_experiment(experiment_name, tags=None):
    if experiment := mlflow.get_experiment_by_name(experiment_name):
        return experiment.experiment_id
    else:
        return mlflow.create_experiment(experiment_name, tags=tags)

def calculate_metrics(model, model_eps, X_gt, S_gt, labels):
    device = model.device
    fpr, tpr, thresholds = roc_curve(labels.ravel(),
                                     torch.abs(model.S).ravel().cpu().numpy())
    precision, recall, thresholds = precision_recall_curve(labels.ravel(),
                                        torch.abs(model.S).ravel().cpu().numpy())
    auc_prc_score = auc(recall, precision)
    auc_roc_score = auc(fpr, tpr)

    metrics = {'AUC_ROC': auc_roc_score,
                'AUC_PRC': auc_prc_score,
                'S_diff': torch.norm(model.S-model_eps.S).cpu().numpy().item(),
                'X_diff': torch.norm(model.X-model_eps.X).cpu().numpy().item(),
                'S_relative_error': (torch.norm(torch.tensor(S_gt, device=device)-model.S).cpu().numpy()/np.linalg.norm(S_gt)).item(),
                'X_relative_error': (torch.norm(torch.tensor(X_gt, device=device)-model.X).cpu().numpy()/np.linalg.norm(X_gt)).item(),
                'S_relative_error_eps': (torch.norm(torch.tensor(S_gt, device=device)-model_eps.S).cpu().numpy()/np.linalg.norm(S_gt)).item(),
                'X_relative_error_eps': (torch.norm(torch.tensor(X_gt, device=device)-model_eps.X).cpu().numpy()/np.linalg.norm(X_gt)).item(),
                }
    metrics['tol'] = metrics['S_relative_error'] + metrics['X_relative_error']
    metrics['diff'] = metrics['S_diff'] + metrics['X_diff']
    return metrics