from pprint import pprint
import os, sys
from copy import deepcopy
from collections import defaultdict
from functools import wraps

import torch
import numpy as np
import pandas as pd
import wandb
import optuna
from optuna.integration.wandb import WeightsAndBiasesCallback
import networkx as nx
from tqdm import tqdm
import geopandas as gpd
from scipy import io
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
from dask.distributed import Client, as_completed
import matplotlib.pyplot as plt


from src.models.lr_ssd.snn__logn_gtv import SNN__LOGN_GTV
from src.models.horpca.horpca_torch import HoRPCA_Singleton
from src.models.lr_ssd.snn_logs import SNN_LOGS
# from src.models.lr_ssd.snn_logn_gtv import SNN__LOGN_GTV

NYC_TAXI_DATA_DIR = os.path.join(os.getcwd(),
                            '..','..','data','nyc_taxi_data')

PERCENTAGES = ['3%', '2%', '1%', '0.7%', '0.3%','0.14%','0.07%','0.014%']
PERCENTAGES = PERCENTAGES[::-1]

class NYCYellowTaxiExperiment:
    def __init__(self, project_name, api_key, exp_config, exp_prefix, group_name, tags,
                **kwargs):
        
        self.project_name = project_name
        self.exp_config = exp_config
        self.group_name = group_name
        self.tags = tags
        wandb.login(key=api_key, verify=True)
        os.environ["WANDB_SILENT"]="true"
        self.exp_prefix = ''
        self.run_kwargs = kwargs
        
        sampler = kwargs.get('sampler', 'TPESampler')
        if sampler == 'TPESampler':
            self.sampler = optuna.samplers.TPESampler()

        study_name = f"{exp_prefix}_{exp_config['name']}"
        self.wandb_kwargs = {'project': project_name,
                        'group': group_name,
                        'tags': tags,
                        'config': exp_config,
                        'name': f"{exp_prefix}_{exp_config['name']}_",
                    }
        
        if exp_config['model'] == 'HoRPCA':
            self.model_objective = HoRPCA_Objective(exp_config, self.wandb_kwargs)
        elif exp_config['model'] == 'SNN_LOGN_GTV':
            self.model_objective = SNN_LOGN_Objective(exp_config, self.wandb_kwargs)
        else:
            raise ValueError("Invalid model name")
        
        self.study = optuna.study.create_study(study_name=study_name,
                                                direction='minimize',
                                                storage='sqlite:///optuna_gorpca_nyc.db',
                                                load_if_exists=True,
                                                sampler=self.sampler)
        self.all_results = defaultdict(list)
        

    def run_study(self, n_trials):
        self.study.optimize(self.model_objective, n_trials=n_trials,)# callbacks=[self.wandbc])
        return self.study.trials_dataframe()
    
class HoRPCA_Objective:
    def __init__(self, config, wandb_kwargs):
        self.data = NYCTaxiDataset()
        self.config = config
        self.wandb_kwargs = wandb_kwargs

    def __call__(self, trial):
        wandb_kwargs = deepcopy(self.wandb_kwargs)
        wandb_kwargs['name'] = self.wandb_kwargs['name'] + f"{trial.number}"
        run = wandb.init(**wandb_kwargs)
        model_control = self.config['model_control']
        search_space = self.config['search_space']
        
        scheme = search_space.get('scheme', 'auto')
        
        if scheme == 'auto':
            ts = np.array([-np.log(trial.suggest_float(f"t_{i}", 0, 1)) for i in range(
                        len(model_control['lr_modes'])+1)])
            ts = list(ts/np.sum(ts))
            alg_params = {'lda_nucs': ts[:-1],
                          'lda1': ts[-1]}
            run.log({f'psi_{m}': ts[i] for i,m in enumerate(model_control['lr_modes'])})
            run.log({'lda': ts[-1]})
            [trial.set_user_attr(f"psi_{m}", ts[i]) for i,m in enumerate(model_control['lr_modes'])];
            trial.set_user_attr('lda', ts[-1])
        elif scheme == 'manual':
            pass
        elif scheme == 'constrained_auto':
            pass
        elif scheme == 'constrained_manual':
            pass
        else:
            raise ValueError(f"Invalid scheme: {scheme}")
        
        alg_params = model_control | alg_params
        model = HoRPCA_Singleton(self.data.dropoffs, **alg_params)
        X, S = model()

        metrics = {'Sparsity': torch.sum(S!=0).cpu().numpy()/S.numel(),
            'converged': model.converged*1, 
            'iterations': model.it, 
            'primal_residual': model.r[-1], 
            'dual_residual': model.s[-1],
            'time': sum(model.times['iteration'])}
        
        abs_s = np.abs(S.cpu().detach().numpy())
        ratios = np.array([0.014, 0.07, 0.14, 0.3, 0.7, 1, 2, 3])/100
        num_detected_events = np.array([sum(self.data.detect_topk_events(abs_s, r)) for r in ratios])
        metrics['BIC_old'], metrics['nonzero_param_old'] = model.bayesian_information_criterion()

        bic_metrics = model.bayesian_information_criterion_modified()
        metrics = metrics|bic_metrics
        for i, pcntg in enumerate(PERCENTAGES):
            metrics[pcntg] = num_detected_events[i]
        run.log(metrics)
        trial.set_user_attr('metrics', metrics)
        run.finish()
        return metrics['BIC']
        
class SNN_LOGN_Objective:
    def __init__(self, config, wandb_kwargs):
        self.data = NYCTaxiDataset()
        self.config = config
        domain = config['model_control'].get('domain','spatial')
        if domain == 'spatial':
            self.G = self.data.G_nyc
        elif domain == 'temporal':
            self.G = self.data.Gt
        elif domain == 'spatio_temporal':
            self.G = nx.cartesian_product(self.data.G_nyc, self.data.Gt)
        self.wandb_kwargs = wandb_kwargs
        
    def __call__(self, trial):
        wandb_kwargs = deepcopy(self.wandb_kwargs)
        wandb_kwargs['name'] = self.wandb_kwargs['name'] + f"_{trial.number}"
        run = wandb.init(**wandb_kwargs)
        model_control = self.config['model_control']
        search_space = self.config['search_space']
        
        scheme = search_space.get('scheme', 'auto')
        
        if scheme == 'auto':
            ts = np.array([-np.log(trial.suggest_float(f"t_{i}", 0, 1)) for i in range(
                        len(model_control['lr_modes'])+1)])
            ts = list(ts/np.sum(ts))
            alg_params = {'psis': ts[:-1],
                          'lda': ts[-1]}
            run.log({f'psi_{m}': ts[i] for i,m in enumerate(model_control['lr_modes'])})
            run.log({'lda': ts[-1]})
            [trial.set_user_attr(f"psi_{m}", ts[i]) for i,m in enumerate(model_control['lr_modes'])];
            trial.set_user_attr('lda', ts[-1])

        elif scheme == 'manual':
            pass
        elif scheme == 'constrained_auto':
            pass
        elif scheme == 'constrained_manual':
            pass
        else:
            raise ValueError(f"Invalid scheme: {scheme}")
        
        domain = model_control.get('domain','spatial')
        if domain == 'spatial':
            model_control['graph_modes'] = [1]
        elif domain == 'temporal':
            model_control['graph_modes'] = [4]
        elif domain == 'spatio_temporal':
            model_control['graph_modes'] = [1,4]

        model = SNN__LOGN_GTV(self.data.dropoffs, self.G, **model_control)

        X, S = model(**(alg_params|model_control))
        
        
        metrics = {
            'Sparsity': torch.sum(model.V.sum(dim=0).coalesce().to_dense()!=0).cpu().numpy()/S.numel(),
            'converged': model.converged*1, 
            'iterations': model.it, 
            'primal_residual': model.r[-1].cpu().item(), 
            'dual_residual': model.s[-1].cpu().item(),
            'time': sum(model.times['iteration'])
        }
        bic_metrics = model.bayesian_information_criterion_modified()
        metrics = metrics|bic_metrics
        metrics['BIC_old'], metrics['nonzero_param_old'] = model.bayesian_information_criterion()
        
        abs_s = np.abs(S.cpu().detach().numpy())
        ratios = np.array([0.014, 0.07, 0.14, 0.3, 0.7, 1, 2, 3])/100
        
        num_detected_events = np.array([sum(self.data.detect_topk_events(abs_s, r)) for r in ratios])
        for i, pcntg in enumerate(PERCENTAGES):
            metrics[pcntg] = num_detected_events[i]
        total_detected = sum(num_detected_events)
        wandb.log(metrics)
        wandb.finish()
        trial.set_user_attr('metrics', metrics)
        # return metrics['BIC']
        return -total_detected




    
class NYCTaxiLR_Sparse_IdStudy:
    def __init__(self, project_name, api_key, exp_config, exp_prefix, group_name, tags, 
                client=Client(n_workers=5),
                **kwargs):
        self.project_name = project_name
        self.exp_config = exp_config
        self.group_name = group_name
        self.tags = tags
        self.all_results = defaultdict(list)
        wandb.login(key=api_key, verify=True)
        self.exp_prefix = exp_prefix
        self.run_kwargs = kwargs
        self.num_active_remote = 0
        self.client = client
        self.study = None

    def run_id_study(self, lr_modes, data='dropoffs'):
        print("Experiment Configuration: ") 
        pprint(self.exp_config)
        models = self.exp_config['models']
        data_variables = self.exp_config.get('data', {'data': data})
        eps = 1/(self.exp_config['independent_var']['t']['resolution']*20)
        t_range = np.linspace(eps, 1,
                              self.exp_config['independent_var']['t']['resolution']+1,
                              endpoint=True)
        

        for key, model_variables in models.items():
            tags = self.tags + [model_variables['name']]
            model_var = deepcopy(model_variables)
            model_var['eps'] = eps

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
                                    name=f"{self.exp_prefix}_{model_variables['name']}_modes_{lr_mode}",
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


def model_runner(model, data_variables, model_variables, step):
    data = get_data(data_variables)
    results = run_model(model, data, model_variables)
    metrics = calculate_metrics(results, data)
    return metrics, step


def run_model(model, data, model_variables):
    if model == 'HoRPCA':
        return run_HoRPCA(data, model_variables)
    elif model == 'SNN_LOGS':
        return run_SNN_LOGS(data, model_variables)
    elif model == 'SNN_LOGN_GTV':
        results = run_SNN_LOGN_GTV(data, model_variables)
        {**results, **(results['model'].calculate_local_variation_measures())}
    else:
        raise ValueError('Invalid model name')

def get_data(data_variables):
    nyc_taxi_dataset = NYCTaxiDataset()

    if data_variables['data'] == 'dropoffs':
        Y = nyc_taxi_dataset.dropoffs
    elif data_variables['data'] == 'pickups':
        Y = nyc_taxi_dataset.pickups
    else:
        raise ValueError("Invalid data variable.")
    
    return {'Y': Y, 'G':nyc_taxi_dataset.G_nyc,
            'dates': nyc_taxi_dataset.dates,
            'regions': nyc_taxi_dataset.regions}


def calculate_metrics(results, data):
    model = results['model']

    metrics = { 'BIC': model.bic[-1].cpu().item(),
                'L1':  torch.sum(torch.abs(model.X)).cpu().item(),
                'S1': torch.sum(torch.abs(model.S)).cpu().item(),
                'nonzero_S': (torch.sum(model.S != 0)/torch.prod(torch.tensor(model.S.shape))).cpu().item(),
                'nonzero_L': (torch.sum(model.X != 0)/torch.prod(torch.tensor(model.S.shape))).cpu().item(),
                'S_fro': torch.norm(model.S).cpu().item(),
                'L_fro': torch.norm(model.X).cpu().item(),
    }
    metrics['primal_residual'] = model.r[-1].cpu().item()
    metrics['dual_residual'] = model.s[-1].cpu().item()

    abs_s = torch.abs(model.S).cpu().numpy()
    ratios = np.array([0.014, 0.07, 0.14, 0.3, 0.7, 1, 2, 3])/100
    num_detected_events = np.array([sum(detect_topk_events(abs_s, r,
                                                           data['dates'],
                                                           data['regions'])) for r in ratios])
    for i, pcntg in enumerate(PERCENTAGES):
        metrics[pcntg] = num_detected_events[i]
    metrics['detected_events'] = num_detected_events
    metrics['Total'] = sum(num_detected_events)
    return metrics


def run_SNN_LOGS(data, variables):
    G = data['G']
    Y = data['Y']
    multipliers = variables.get('multipliers', defaultdict(lambda x: 1))
    t = variables['t']
    eps = variables['eps']
    var1 = {}
    var1['lr_modes'] = variables['lr_modes']
    var1['graph_modes'] = variables['graph_modes']
    var1['grouping'] = variables['grouping']
    var1['weighing'] = variables['weighing']
    var1['r_hop'] = variables['r_hop']
    var1['device'] = variables.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    var1['dtype'] = variables.get('dtype', torch.float64)
    var1['verbose'] = variables.get('verbose', 0)
    model1 = SNN_LOGS(Y, G, **var1)
    model2 = SNN_LOGS(Y, G, **var1)
    var2 = {}
    var2['psis'] = [1-t]*len(variables['lr_modes'])
    var1['psis'] = [1-t]*len(variables['lr_modes'])
    var1['psis'] = [var1['psis'][i]*multipliers[f'psi_{m}'] for i,m in enumerate(var1['lr_modes'])]
    var2['lda'] = t*multipliers['lambda']
    var2['max_iter'] = variables.get('max_iter', 150)
    var2['rho'] = variables.get('rho', 4*np.abs(Y).sum()/Y.size)
    var2['err_tol'] = variables.get('err_tol', 1e-6)
    var2['rho_update'] = variables.get('rho_update', 1)
    var2['rho_update_thr'] = variables.get('rho_update_thr', 100)
    X1, S1 = model1(**var2)
    var2['psis'] = [1-t+eps]*len(variables['lr_modes'])
    var2['psis'] = [var2['psis'][i]*multipliers[f'psi_{m}'] for i, m in enumerate(var1['lr_modes'])]
    # var2['psis'] = [var2['psis'][m-1]*multipliers[f'psi_{m}'] for i, m in var1['lr_modes']]
    var2['lda'] = (t - eps)*multipliers['lambda']
    X2, S2 = model2(**var2)
    return {'model': model1, 'model_eps': model2}


def run_HoRPCA(data, variables):
    Y = data['Y']
    var = {}
    multipliers = variables.get('multipliers', defaultdict(lambda x: 1))
    t = variables['t']
    
    eps = variables['eps']
    var['lr_modes'] = variables['lr_modes']
    var['lda1'] = t*multipliers['lambda']
    var['lda_nucs'] = [1-t]*len(variables['lr_modes'])
    var['lda_nucs'] = [var['lda_nucs'][m-1]*multipliers[f'psi_{m}'] for m in var['lr_modes']]
    var['rho'] = variables.get('rho', 4*np.abs(Y).sum()/Y.size)
    var['err_tol'] = variables.get('err_tol', 1e-6)
    var['maxit'] = variables.get('maxit', 300)
    var['step_size_growth'] = variables.get('step_size_growth', variables.get('rho_update', 1))
    var['mu'] = variables.get('mu', variables.get('rho_update_thr', 100))
    var['verbose'] = variables.get('verbose', 0)
    var['device'] = variables.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    var['dtype'] = variables.get('dtype', torch.float64)
    # print(f"Running HoRPCA with Model Variables")
    # pprint(var)
    model1 = HoRPCA_Singleton(Y, **var)
    X1,S1 = model1()
    
    var['lda1'] = t - eps
    var['lda1'] = t*multipliers['lambda']
    var['lda_nucs'] = [1-t+eps]*len(variables['lr_modes'])
    var['lda_nucs'] = [var['lda_nucs'][m-1]*multipliers[f'psi_{m}'] for m in var['lr_modes']]
    model2 = HoRPCA_Singleton(Y, **var)
    X2,S2 = model2()
    return {'model': model1, 'model_eps': model2}


def run_SNN_LOGN_GTV(data, variables):
    G = data['G']
    Y = data['Y']
    
    t = variables['t']
    eps = variables['eps']

    var = {}
    var['lr_modes'] = variables['lr_modes']
    var['graph_modes'] = variables['graph_modes']
    var['grouping'] = variables['grouping']
    var['weighing'] = variables['weighing']
    var['r_hop'] = variables['r_hop']
    var['device'] = variables.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    var['dtype'] = variables.get('dtype', torch.float64)
    var['verbose'] = variables.get('verbose', 0)
    var['gtvr_config'] = variables['gtvr_config']

    model1 = SNN__LOGN_GTV(Y, G, **var)
    model2 = SNN__LOGN_GTV(Y, G, **var)
    var2 = {}
    var2['psis'] = [1-t]*len(variables['lr_modes'])
    var2['lda'] = t
    var2['lda_gtvs'] = variables['lda_gtvs']
    var2['max_iter'] = variables.get('max_iter', 150)
    var2['rho'] = variables.get('rho', 4*np.abs(Y).sum()/Y.size)
    var2['err_tol'] = variables.get('err_tol', 1e-6)
    var2['rho_update'] = variables.get('rho_update', 1)
    var2['rho_update_thr'] = variables.get('rho_update_thr', 100)
    X1, S1 = model1(**var2)
    var2['psis'] = [1-t+eps]*len(variables['lr_modes'])
    var2['lda'] = t - eps
    X2, S2 = model2(**var2)
    return {'model': model1, 'model_eps': model2}


class NYCTaxiDataset:
    def __init__(self, data_dir=NYC_TAXI_DATA_DIR):
        self.zones = gpd.read_file(os.path.join(data_dir,'taxi_zones_shapefile','taxi_zones.shp'))
        self.zone_lookup = pd.read_csv(
            os.path.join(data_dir, 'taxi_zone_lookup.csv')
        )

        # # Load Emre's settings
        self.dates = io.loadmat(os.path.join(data_dir,'dates.mat'))
        self.regions = io.loadmat(os.path.join(data_dir,'regions.mat'))
        self.neighbors = io.loadmat(os.path.join(data_dir,'neighbors.mat'))
        self.regions = self.regions['regions'].ravel()
        
        self.zones = self.zones.iloc[self.regions-1].reset_index()
        self.arrivals = io.loadmat(os.path.join(data_dir,'arrivals.mat'))['Y']
        self.dropoffs = np.moveaxis(self.arrivals,[0,1,2,3],[3,2,1,0])
        self.pos = np.zeros((81,2))
        self.pos[:,0] = self.zones.geometry.centroid.x.values
        self.pos[:,1] = self.zones.geometry.centroid.y.values
        edge_list = nx.from_scipy_sparse_array(
            kneighbors_graph(self.pos, 2, mode='connectivity', include_self=False)).edges()

        adjacency_list = []
        for index, polygon in self.zones.iterrows():
            for other_index, other_polygon in self.zones.iterrows():
                if index != other_index and polygon.geometry.intersects(other_polygon.geometry.buffer(1)):
                    adjacency_list.append((index, other_index))

        self.G_nyc = nx.Graph()
        self.G_nyc.add_nodes_from(range(81))
        self.G_nyc.add_edges_from(adjacency_list)
        self.G_nyc.add_edges_from([(11,19), (36,42), (67,77), (19,67)])
        self.G_nyc.add_edges_from(edge_list)
        self.Gt = nx.grid_graph(dim=(24, ), periodic=False)


    def plot_zones(self, **kwargs):
        fig, ax = plt.subplots(1,1, figsize=kwargs.get('figsize', (10,16)));
        self.zones.geometry.buffer(-100).plot(ax = ax)
        ax.set_title('NYC Taxi Zones')
        self.zones.plot(ax=ax, color='brown', edgecolor='black');
        nx.draw(self.G_nyc, 
                pos={list(self.G_nyc)[i]: self.pos[i,:] for i in range(len(self.G_nyc))},
                ax=ax, 
                node_size = kwargs.get('node_size', 100),
                edge_color = kwargs.get('edge_color', 'black'),
                with_labels = kwargs.get('with_labels', True),
                node_color = kwargs.get('node_color', 'C3'),
                font_size = kwargs.get('font_size', 9),);

    def detect_topk_events(self, anomaly_scores, ratio):
        """Detect the events that are in the top-k of the anomaly scores.

        Parameters:
        ----------
            anomaly_scores (np.ndarray): 81x53x7x24 array of anomaly scores.
            ratio (float): Top-r ratio of the entries of the anomaly scores to consider.
        
        Returns:
        -------
            detected_events (np.ndarray): 20x1 array of detected events.
        """
        events_start_ts = pd.to_datetime(['01-Jan-2018', '03-Jan-2018 16:00:00', '14-Jan-2018 09:00:00', '20-Jan-2018 08:00:00', 
                                        '4-Mar-2018 15:00:00', '08-Mar-2018 18:00:00', '17-Mar-2018 11:00:00', '20-Mar-2018 10:00:00',
                                        '21-Mar-2018 16:00:00', '01-Jul-2018 17:00:00', '04-Jul-2018 17:00:00', '25-Sep-2018 10:00:00',
                                        '04-Oct-2018 08:00:00', '04-Nov-2018 12:00:00', '09-Nov-2018 19:00:00', '22-Nov-2018 21:00:00',
                                        '4-Dec-2018 19:00:00', '16-Dec-2018 10:00:00', '28-Dec-2018 12:00:00', '31-Dec-2018 21:00:00',
                                        ], format='mixed')
        events_end_ts = pd.to_datetime(['01-Jan-2018 02:00:00', '03-Jan-2018 22:00:00', '14-Jan-2018 17:00:00', '20-Jan-2018 15:00:00',
                                '4-Mar-2018 22:00:00', '08-Mar-2018 23:59:00', '17-Mar-2018 17:00:00', '20-Mar-2018 20:00:00',
                                '21-Mar-2018 22:00:00', '01-Jul-2018 22:00:00', '04-Jul-2018 23:00:00', '25-Sep-2018 20:00:00',
                                '04-Oct-2018 15:00:00', '04-Nov-2018 17:00:00', '09-Nov-2018 23:30:00', '22-Nov-2018 23:59:00',
                                '4-Dec-2018 23:59:00', '16-Dec-2018 15:00:00', '28-Dec-2018 17:00:00', '31-Dec-2018 23:59:00'
                                    ], format='mixed')
        indd = np.flip(np.argsort(anomaly_scores, axis=None))
        ind = np.unravel_index(indd[:int(len(indd)*ratio)], anomaly_scores.shape)
        topk_event_idx = ind
        anomaly_mask = np.zeros(anomaly_scores.shape, dtype=bool)
        anomaly_mask[topk_event_idx] =1
    #     num_detected_events = 0
        detected_events = np.zeros(20)

        idxs = np.arange(81)
        # w = events_start_ts.isocalendar().week
        # d = events_start_ts.day_of_week
        doy = events_start_ts.day_of_year
        w = (doy-1)//(7)
        d = (doy-1) % 7
        h_s = events_start_ts.hour
        h_e = events_end_ts.hour
        for i in range(20):
            event_mask = np.zeros(anomaly_scores.shape, dtype=bool)
            locations = self.dates['dates'][2].ravel()[i].ravel()
            
            for loc in locations: 
                # event_mask[idxs[regions==loc], w[i]-1, d[i], h_s[i]:h_e[i]] = 1
                # event_mask[idxs[regions==loc], w[i]-1, d[i], h_e[i]] = 1
                event_mask[idxs[self.regions==loc], w[i], d[i], h_s[i]:h_e[i]] = 1
                event_mask[idxs[self.regions==loc], w[i], d[i], h_e[i]] = 1
            if np.any(event_mask * anomaly_mask):
    #             num_detected_events +=1
                detected_events[i]=1
        return detected_events

def detect_topk_events(anomaly_scores, ratio, dates, regions):
    """Detect the events that are in the top-k of the anomaly scores.

    Parameters:
    ----------
        anomaly_scores (np.ndarray): 81x53x7x24 array of anomaly scores.
        ratio (float): Top-r ratio of the entries of the anomaly scores to consider.
    
    Returns:
    -------
        detected_events (np.ndarray): 20x1 array of detected events.
    """
    events_start_ts = pd.to_datetime(['01-Jan-2018', '03-Jan-2018 16:00:00', '14-Jan-2018 09:00:00', '20-Jan-2018 08:00:00', 
                                    '4-Mar-2018 15:00:00', '08-Mar-2018 18:00:00', '17-Mar-2018 11:00:00', '20-Mar-2018 10:00:00',
                                    '21-Mar-2018 16:00:00', '01-Jul-2018 17:00:00', '04-Jul-2018 17:00:00', '25-Sep-2018 10:00:00',
                                    '04-Oct-2018 08:00:00', '04-Nov-2018 12:00:00', '09-Nov-2018 19:00:00', '22-Nov-2018 21:00:00',
                                    '4-Dec-2018 19:00:00', '16-Dec-2018 10:00:00', '28-Dec-2018 12:00:00', '31-Dec-2018 21:00:00',
                                    ], format='mixed')
    events_end_ts = pd.to_datetime(['01-Jan-2018 02:00:00', '03-Jan-2018 22:00:00', '14-Jan-2018 17:00:00', '20-Jan-2018 15:00:00',
                             '4-Mar-2018 22:00:00', '08-Mar-2018 23:59:00', '17-Mar-2018 17:00:00', '20-Mar-2018 20:00:00',
                            '21-Mar-2018 22:00:00', '01-Jul-2018 22:00:00', '04-Jul-2018 23:00:00', '25-Sep-2018 20:00:00',
                            '04-Oct-2018 15:00:00', '04-Nov-2018 17:00:00', '09-Nov-2018 23:30:00', '22-Nov-2018 23:59:00',
                            '4-Dec-2018 23:59:00', '16-Dec-2018 15:00:00', '28-Dec-2018 17:00:00', '31-Dec-2018 23:59:00'
                                ], format='mixed')
    indd = np.flip(np.argsort(anomaly_scores, axis=None))
    ind = np.unravel_index(indd[:int(len(indd)*ratio)], anomaly_scores.shape)
    topk_event_idx = ind
    anomaly_mask = np.zeros(anomaly_scores.shape, dtype=bool)
    anomaly_mask[topk_event_idx] =1
#     num_detected_events = 0
    detected_events = np.zeros(20)

    idxs = np.arange(81)
    # w = events_start_ts.isocalendar().week
    # d = events_start_ts.day_of_week
    doy = events_start_ts.day_of_year
    w = (doy-1)//(7)
    d = (doy-1) % 7
    h_s = events_start_ts.hour
    h_e = events_end_ts.hour
    for i in range(20):
        event_mask = np.zeros(anomaly_scores.shape, dtype=bool)
        locations = dates['dates'][2].ravel()[i].ravel()
        
        for loc in locations: 
            # event_mask[idxs[regions==loc], w[i]-1, d[i], h_s[i]:h_e[i]] = 1
            # event_mask[idxs[regions==loc], w[i]-1, d[i], h_e[i]] = 1
            event_mask[idxs[regions==loc], w[i], d[i], h_s[i]:h_e[i]] = 1
            event_mask[idxs[regions==loc], w[i], d[i], h_e[i]] = 1
        if np.any(event_mask * anomaly_mask):
#             num_detected_events +=1
            detected_events[i]=1
    return detected_events#num_detected_events

METRICS = ['Total',
            'S_diff', 'L_diff', 
           'L_nuc_1', 'L_nuc_2', 'L_nuc_3', 'L_nuc_4',
           'S_nuc_1', 'S_nuc_2', 'S_nuc_3', 'S_nuc_4', 
           'L_nuc_1_eps','L_nuc_2_eps','L_nuc_3_eps', 'L_nuc_4_eps',
           'S_nuc_1_eps', 'S_nuc_2_eps', 'S_nuc_3_eps', 'S_nuc_4_eps',
           'L1', 'L1_eps', 'S1', 'S1_eps',
           'nonzero_S', 'nonzero_L',
           'ranks_S1', 'ranks_S2', 'ranks_S3', 'ranks_S4',
           'ranks_L1', 'ranks_L2', 'ranks_L3', 'ranks_L4',
           'S_fro', 'L_fro',
           'L_fro_eps', 'S_fro_eps',
           'S_err', 'L_err',
           'tol' ,'diff',
           'primal_residual', 'dual_residual', 'objective_value']