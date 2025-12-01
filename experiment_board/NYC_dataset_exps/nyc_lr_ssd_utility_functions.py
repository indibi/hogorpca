from pprint import pprint
import sys
import os

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import networkx as nx
from scipy import io
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph

module_path = os.path.abspath(os.path.join('..','..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from src.proximal_ops.prox_overlapping_grouped_l21 import group_indicator_matrix


ALGORITHMS = ['Trend', 'Aniso_Trend', 'Grouped_Aniso', 'Aniso_Trend', 'Grouped_Aniso_Trend']
PERCENTAGES = ['3.0%','2.0%','1.0%','0.7%','0.3%','0.14%','0.07%','0.014%']


def champion_callback(study, frozen_trial):
    winner = study.user_attrs.get('winner', None)  # old winner trial if exists
    best_trial = max(study.best_trials, key= lambda t: t.values[0])
    if study.best_trials and winner != best_trial.values:
        study.set_user_attr("winner", best_trial.values)
        if winner:
            print(f"Algorithm: {frozen_trial.user_attrs['algorithm']}\t Trial {frozen_trial.number} achieved value: {frozen_trial.values}.")
            pprint([frozen_trial.user_attrs[pcntg] for pcntg in PERCENTAGES])
            pprint(frozen_trial.user_attrs['metrics'])
            pprint(frozen_trial.params)
        else:
            print(f"Algorithm: {frozen_trial.user_attrs['algorithm']}\t Trial {frozen_trial.number} achieved value: {frozen_trial.values}.")
            pprint([frozen_trial.user_attrs[pcntg] for pcntg in PERCENTAGES])
            pprint(frozen_trial.user_attrs['metrics'])
            pprint(frozen_trial.params)

def run_study(algo, device, maxit, n_trials):
    cwd = os.getcwd()
    data_dir = os.path.join(cwd,'..','..','data','nyc_taxi_data')

    zones = gpd.read_file(os.path.join(data_dir,'taxi_zones_shapefile','taxi_zones.shp'))
    zone_lookup = os.path.join(data_dir, 'taxi_zone_lookup.csv')
    zone_lookup = pd.read_csv(zone_lookup)
    # # Load Emre's settings
    dates = io.loadmat(os.path.join(data_dir,'dates.mat'))
    regions = io.loadmat(os.path.join(data_dir,'regions.mat'))
    neighbors = io.loadmat(os.path.join(data_dir,'neighbors.mat'))
    regions=regions['regions'].ravel()
    zones = zones.iloc[regions-1]
    zones2 = zones.reset_index()
    arrivals = io.loadmat('arrivals.mat')['Y']
    dropoffs = np.moveaxis(arrivals,[0,1,2,3],[3,2,1,0])

    pos = np.zeros((81,2))
    pos[:,0] = zones.geometry.centroid.x.values
    pos[:,1] = zones.geometry.centroid.y.values
    edge_list = nx.from_scipy_sparse_array( kneighbors_graph(pos, 2, mode='connectivity', include_self=False)).edges()

    adjacency_list = []
    for index, polygon in zones2.iterrows():
        for other_index, other_polygon in zones2.iterrows():
            if index != other_index and polygon.geometry.intersects(other_polygon.geometry.buffer(1)):
                adjacency_list.append((index, other_index))

    G_nyc = nx.Graph()
    G_nyc.add_nodes_from(range(81))
    G_nyc.add_edges_from(adjacency_list)
    G_nyc.add_edges_from([(11,19), (36,42), (67,77), (19,67)])
    # G_nyc.add_edges_from(edge_list)
    # G_nyc =nx.DiGraph(G_nyc)
    LG = nx.line_graph(G_nyc)
    # LG = nx.DiGraph(LG)

    # fig, axe = draw_graph_signal(G_nyc, np.ones(81), pos=pos, suptitle='NYC Taxi Zones', node_size=100, cmap='viridis', figsize=(6,6))
    fig, ax = plt.subplots(1,1, figsize=(10,16));
    zones.geometry.buffer(-100).plot(ax = ax)
    Gt = nx.path_graph(24)

    dtype = torch.float64
    Y = torch.tensor(dropoffs, dtype=dtype, device=device)
    G_ind, w1 = group_indicator_matrix(G_nyc, grouping='neighbor', weighing='size_normalized', device='cpu')
    G_ind = G_ind.to(device=device, dtype=dtype)
    w1 = w1.to(device=device, dtype=dtype)
    # G_ind2 = #G_ind.clone().detach().t().to_sparse_csr()

    G_ind_coo = G_ind.to_sparse_coo();
    G_ind_T = G_ind.clone().detach().t().to_sparse_csc(); 
    B = nx.incidence_matrix(G_nyc, oriented=True)
    B1 = torch.sparse_csr_tensor(B.indptr, B.indices, B.data, device=device, dtype=dtype)
    B = nx.incidence_matrix(LG, oriented=True)
    B2 = torch.sparse_csr_tensor(B.indptr, B.indices, B.data, device=device, dtype=dtype)
    B2 = B1.t()@B2.t().to_sparse_csr()
    B1 = B1.t().to_sparse_csr()
    B = nx.incidence_matrix(Gt, oriented=True)
    Bt = torch.sparse_csr_tensor(B.indptr, B.indices, B.data, device=device, dtype=dtype).t().to_sparse_csr()

    G_ind2 = B1.clone().detach().to_sparse_csr()
    G_ind2_coo = G_ind2.to_sparse_coo()
    G_ind2_T = G_ind2.clone().detach().t().to_sparse_csc()
    w2 = torch.sum(G_ind2, dim=1, keepdim=True)

    maxit = 150
    err_tol = 1e-5

    

    def objective_f(trial, algo=algos[0], B1_=B1, B2_=B2, Bt_=Bt, G_ind_=[G_ind, G_ind2], G_ind_coo_=[G_ind_coo, G_ind2_coo], G_ind_T_=[G_ind_T, G_ind2_T]):
        # ['Trend', 'Aniso_Trend', 'Grouped_Aniso', 'Aniso_Trend', 'Grouped_Aniso_Trend']
        if algo == 'Trend':
            alg_parameters={'psis': [trial.suggest_float('psi_1', 1e-2, 1e2, log=True),
                                    trial.suggest_float('psi_2', 1e-2, 1e2, log=True),
                                    trial.suggest_float('psi_3', 1e-2, 1e2, log=True),
                                    trial.suggest_float('psi_4', 1e-2, 1e2, log=True)],
                            'lda_0': trial.suggest_float('lda_0', 1e-4, 1e3, log=True),
                            'lda_g0': 0,
                            'lda_1': 0,
                            'lda_g1': 0,
                            'lda_2' : trial.suggest_float('lda_2', 1e-4, 1e3, log=True),
                            'lda_t' : trial.suggest_float('lda_t', 1e-4, 1e3, log=True),
                            'lda_f': 1000,
                            'rho': trial.suggest_float('rho', 1e-4, 1e3, log=True),
                            'max_iter': maxit,
                            }
        elif algo == 'Grouped':
            alg_parameters={'psis': [trial.suggest_float('psi_1', 1e-2, 1e2, log=True),
                                    trial.suggest_float('psi_2', 1e-2, 1e2, log=True),
                                    trial.suggest_float('psi_3', 1e-2, 1e2, log=True),
                                    trial.suggest_float('psi_4', 1e-2, 1e2, log=True)],
                            'lda_0': 0,
                            'lda_g0': trial.suggest_float('lda_g0', 1e-3, 1e3, log=True),
                            'lda_1': 0,
                            'lda_g1': 0,
                            'lda_2' : 0,
                            'lda_t' : trial.suggest_float('lda_t', 1e-4, 1e3, log=True),
                            'lda_f': 1000,
                            'rho': trial.suggest_float('rho', 1e-4, 1e3, log=True),
                            'max_iter': maxit,
                            }
        elif algo == 'Aniso_Trend':
            alg_parameters={'psis': [trial.suggest_float('psi_1', 1e-2, 1e2, log=True),
                                    trial.suggest_float('psi_2', 1e-2, 1e2, log=True),
                                    trial.suggest_float('psi_3', 1e-2, 1e2, log=True),
                                    trial.suggest_float('psi_4', 1e-2, 1e2, log=True)],
                            'lda_0': trial.suggest_float('lda_0', 1e-4, 1e3, log=True),
                            'lda_g0': 0,
                            'lda_1': trial.suggest_float('lda_1', 1e-4, 1e3, log=True),
                            'lda_g1': 0,
                            'lda_2' : trial.suggest_float('lda_2', 1e-4, 1e3, log=True),
                            'lda_t' : trial.suggest_float('lda_t', 1e-4, 1e3, log=True),
                            'lda_f': 1000,
                            'rho': trial.suggest_float('rho', 1e-4, 1e3, log=True),
                            'max_iter': maxit,
                            }
        elif algo == 'Aniso':
            alg_parameters={'psis': [trial.suggest_float('psi_1', 1e-2, 1e2, log=True),
                                    trial.suggest_float('psi_2', 1e-2, 1e2, log=True),
                                    trial.suggest_float('psi_3', 1e-2, 1e2, log=True),
                                    trial.suggest_float('psi_4', 1e-2, 1e2, log=True)],
                            'lda_0': trial.suggest_float('lda_0', 1e-5, 1.0, log=True),
                            'lda_g0': 0,
                            'lda_1': trial.suggest_float('lda_1', 1e-5, 1.0, log=True),
                            'lda_g1': 0,
                            'lda_2' : 0,
                            'lda_t' : trial.suggest_float('lda_t', 1e-5, 1.0, log=True),
                            'lda_f': 1000,
                            'rho': trial.suggest_float('rho', 1e-5, 1e1, log=True),
                            'max_iter': maxit,
                            }
        elif algo == 'Grouped_Aniso_Trend':
            alg_parameters={'psis': [trial.suggest_float('psi_1', 1e-2, 1e2, log=True),
                                    trial.suggest_float('psi_2', 1e-2, 1e2, log=True),
                                    trial.suggest_float('psi_3', 1e-2, 1e2, log=True),
                                    trial.suggest_float('psi_4', 1e-2, 1e2, log=True)],
                            'lda_0': trial.suggest_float('lda_0', 1e-5, 1.0, log=True),
                            'lda_g0': trial.suggest_float('lda_g0', 1e-5, 1.0, log=True),
                            'lda_1': trial.suggest_float('lda_1', 1e-5, 1.0, log=True),
                            'lda_g1': 0,
                            'lda_2' : trial.suggest_float('lda_2', 1e-5, 1.0, log=True),
                            'lda_t' : trial.suggest_float('lda_t', 1e-5, 1.0, log=True),
                            'lda_f': 1000,
                            'rho': trial.suggest_float('rho', 1e-5, 1e1, log=True),
                            'max_iter': maxit,
                            }
        elif algo == 'Grouped_Aniso':
            alg_parameters={'psis': [trial.suggest_float('psi_1', 1e-2, 1e2, log=True),
                                    trial.suggest_float('psi_2', 1e-2, 1e2, log=True),
                                    trial.suggest_float('psi_3', 1e-2, 1e2, log=True),
                                    trial.suggest_float('psi_4', 1e-2, 1e2, log=True)],
                            'lda_0': trial.suggest_float('lda_0', 1e-5, 1.0, log=True),
                            'lda_g0': trial.suggest_float('lda_g0', 1e-5, 1.0, log=True),
                            'lda_1': trial.suggest_float('lda_1', 1e-5, 1.0, log=True),
                            'lda_g1': 0,
                            'lda_2' : 0,
                            'lda_t' : trial.suggest_float('lda_t', 1e-5, 1.0, log=True),
                            'lda_f': 1000,
                            'rho': trial.suggest_float('rho', 1e-5, 1e1, log=True),
                            'max_iter': maxit,
                            }
        elif algo == 'Grouped_Iso_Trend':
            alg_parameters={'psis': [trial.suggest_float('psi_1', 1e-2, 1e2, log=True),
                                    trial.suggest_float('psi_2', 1e-2, 1e2, log=True),
                                    trial.suggest_float('psi_3', 1e-2, 1e2, log=True),
                                    trial.suggest_float('psi_4', 1e-2, 1e2, log=True)],
                            'lda_0': trial.suggest_float('lda_0', 1e-5, 1.0, log=True),
                            'lda_g0': trial.suggest_float('lda_g0', 1e-5, 1.0, log=True),
                            'lda_1': 0,
                            'lda_g1': trial.suggest_float('lda_g1', 1e-5, 1.0, log=True),
                            'lda_2' : trial.suggest_float('lda_2', 1e-5, 1.0, log=True),
                            'lda_t' : trial.suggest_float('lda_t', 1e-5, 1.0, log=True),
                            'lda_f': 1000,
                            'rho': trial.suggest_float('rho', 1e-5, 1e1, log=True),
                            'max_iter': maxit,
                            }


        lr_ssd = LR_SSD(Y, B1_, B2_, Bt_, lr_modes=[1,2,3,4], time_mode=4, graph_modes=[1],
                G_ind=G_ind_, G_ind_coo=G_ind_coo_, G_ind_T=G_ind_T_,
                group_weights=[w1,w2], rho_update_thr=100, verbose=0)
        X, S = lr_ssd(**alg_parameters);

        metrics = {'Sparsity': torch.sum(S!=0).cpu().numpy()/S.numel(),
                'converged': lr_ssd.converged*1, 
                'iterations': lr_ssd.it, 
                'primal_residual': lr_ssd.r[-1].cpu().numpy(), 
                'dual_residual': lr_ssd.s[-1].cpu().numpy(),
                'time': sum(lr_ssd.times['iteration'])}

        trial.set_user_attr('algorithm', algo)
        trial.set_user_attr('metrics', metrics)
        abs_s = np.abs(S.cpu().detach().numpy())
        ratios = np.array([0.014, 0.07, 0.14, 0.3, 0.7, 1, 2, 3])/100
        num_detected_events = np.array([sum(detect_topk_events(abs_s, r)) for r in ratios])
        for i, pcntg in enumerate(prcntgs):
            trial.set_user_attr(pcntg, num_detected_events[i])
        return sum(num_detected_events), num_detected_events[-1]
    
    objective = partial(objective_f, algo=algo, B1_=B1, B2_=B2, Bt_=Bt, G_ind_=[G_ind, G_ind2], G_ind_coo_=[G_ind_coo, G_ind2_coo], G_ind_T_=[G_ind_T, G_ind2_T])
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    study = optuna.create_study(directions=['maximize', 'maximize'])
    study.optimize(objective, n_trials=n_trials, callbacks=[champion_callback])


def detect_topk_events(anomaly_scores, ratio):
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