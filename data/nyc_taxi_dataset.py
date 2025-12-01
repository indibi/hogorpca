import os, sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
import geopandas as gpd
from scipy import io
from sklearn.neighbors import kneighbors_graph
import matplotlib.pyplot as plt

DATA_DIR = Path(os.path.join(os.path.dirname(__file__)))
BASE_DIR = DATA_DIR.parent
NYC_TAXI_DATA_DIR = DATA_DIR / 'nyc_taxi_data'
sys.path.append(BASE_DIR.as_posix())


from src.multilinear_ops.matricize import matricize
from src.multilinear_ops.tensorize import tensorize


PERCENTAGES = ['3%', '2%', '1%', '0.7%', '0.3%','0.14%','0.07%','0.014%']
PERCENTAGES = PERCENTAGES[::-1]
RATIOS = [0.03, 0.02, 0.01, 0.007, 0.003, 0.0014, 0.0007, 0.00014][::-1]


class NYCTaxiDataset:
    def __init__(self, data_dir=NYC_TAXI_DATA_DIR, **kwargs):
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
        self._load_event_labels()
        # self._standardize(kwargs.get('standardize', True))

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
        
        self._calculate_edge_weights(**kwargs.get('edge_weighting', {}))

        self.Gt = nx.grid_graph(dim=(24, ), periodic=False)
        self.G = self.G_nyc
        self.Y = self.dropoffs
    
    # def _standardize(self, standardize):

    def _calculate_edge_weights(self, method=None, **kwargs):
        if method is not None:
            nodelist = list(self.G_nyc.nodes())
            edgelist = list(self.G_nyc.edges())
            B = nx.incidence_matrix(self.G_nyc,
                                        nodelist=nodelist, edgelist=edgelist, oriented=False).T
            diffs = B @ matricize(self.dropoffs, [1])
            if method == 'rbf':
                c = kwargs.get('c', 0.05)
                sigma_0 = np.sum(self.dropoffs**2)
                weights = -(diffs**2).sum(axis=1) / (2 * sigma_0*c)
                weights = np.exp(weights)
            if method == 'median':
                c = kwargs.get('c', 1.0)
                weights = np.median(np.abs(diffs), axis=1)
                weights = c/(weights + 1e-6)
            if method == 'quantile':
                q = kwargs.get('q', 0.75)
                c = kwargs.get('c', 1.0)
                weights = np.quantile(np.abs(diffs), q, axis=1)
                weights = c/(weights + 1e-6)
            for i, edge in enumerate(edgelist):
                self.G_nyc.edges[edge]['weight'] = weights[i]
    
    def _calculate_mode_wise_means(self, Y):
        """Calculate the mean for each mode of the tensor."""
        mode_wise_means = []
        for i in range(4):
            Ym = matricize(Y, [i+1])
            mode_mean = np.mean(Ym, axis=1)
            mode_wise_means.append(mode_mean)
        return mode_wise_means



    def plot_zones(self, **kwargs):
        ax = kwargs.get('ax', None)
        # fig = kwargs.get('fig', None)
        if ax is None:
            fig, ax = plt.subplots(1,1, figsize=kwargs.get('figsize', (10,16)));
        self.zones.geometry.buffer(-100).plot(ax = ax)
        ax.set_title('NYC Taxi Zones')
        self.zones.plot(ax=ax,
                        color=kwargs.get('zone_color', 'brown'),
                        cmap=kwargs.get('zone_cmap', None),
                        vmin=kwargs.get('zone_vmin', None),
                        vmax=kwargs.get('zone_vmax', None),
                        edgecolor='black');
        nx.draw(self.G_nyc, 
                pos={list(self.G_nyc)[i]: self.pos[i,:] for i in range(len(self.G_nyc))},
                ax=ax,
                alpha = kwargs.get('graph_alpha', 1.0),
                node_size = kwargs.get('node_size', 100),
                edge_color = kwargs.get('edge_color', 'black'),
                with_labels = kwargs.get('with_labels', True),
                node_color = kwargs.get('node_color', 'C3'),
                font_size = kwargs.get('font_size', 9),
                cmap = kwargs.get('cmap', None),
                vmin = kwargs.get('vmin', None),
                vmax = kwargs.get('vmax', None),
                );

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
    
    def _load_event_labels(self):
        self.events_start_ts = pd.to_datetime(['01-Jan-2018', '03-Jan-2018 16:00:00', '14-Jan-2018 09:00:00', '20-Jan-2018 08:00:00', 
                                        '4-Mar-2018 15:00:00', '08-Mar-2018 18:00:00', '17-Mar-2018 11:00:00', '20-Mar-2018 10:00:00',
                                        '21-Mar-2018 16:00:00', '01-Jul-2018 17:00:00', '04-Jul-2018 17:00:00', '25-Sep-2018 10:00:00',
                                        '04-Oct-2018 08:00:00', '04-Nov-2018 12:00:00', '09-Nov-2018 19:00:00', '22-Nov-2018 21:00:00',
                                        '4-Dec-2018 19:00:00', '16-Dec-2018 10:00:00', '28-Dec-2018 12:00:00', '31-Dec-2018 21:00:00',
                                        ], format='mixed')
        self.events_end_ts = pd.to_datetime(['01-Jan-2018 02:00:00', '03-Jan-2018 22:00:00', '14-Jan-2018 17:00:00', '20-Jan-2018 15:00:00',
                                '4-Mar-2018 22:00:00', '08-Mar-2018 23:59:00', '17-Mar-2018 17:00:00', '20-Mar-2018 20:00:00',
                                '21-Mar-2018 22:00:00', '01-Jul-2018 22:00:00', '04-Jul-2018 23:00:00', '25-Sep-2018 20:00:00',
                                '04-Oct-2018 15:00:00', '04-Nov-2018 17:00:00', '09-Nov-2018 23:30:00', '22-Nov-2018 23:59:00',
                                '4-Dec-2018 23:59:00', '16-Dec-2018 15:00:00', '28-Dec-2018 17:00:00', '31-Dec-2018 23:59:00'
                                    ], format='mixed')

        self.event_d_o_ys = self.events_start_ts.day_of_year
        
        idxs = np.arange(81)
        self.event_weeks = (self.event_d_o_ys-1)//(7)
        self.event_days = (self.event_d_o_ys-1) % 7
        self.event_hour_s = self.events_start_ts.hour
        self.event_hour_e = self.events_end_ts.hour

    def _get_event_mask(self, event_number):
        event_mask = np.zeros(self.dropoffs.shape, dtype=bool)
        locations = self.dates['dates'][2].ravel()[event_number].ravel()
        
        w = self.event_weeks[event_number]
        d = self.event_days[event_number]
        h_s = self.event_hour_s[event_number]
        h_e = self.event_hour_e[event_number]
        idxs = np.arange(81)

        for loc in locations:
            event_mask = np.zeros(self.dropoffs.shape, dtype=bool)
            event_mask[idxs[self.regions==loc], w, d, h_s:h_e] = 1
            event_mask[idxs[self.regions==loc], w, d, h_e] = 1
        return event_mask


    def anomaly_detection_score(self, triggered_labels,
                                keys=['num_detected_events', 'detected_events','total_detected_support_ratio']):
        total_mask = np.zeros_like(triggered_labels, dtype=bool)
        D = self.dropoffs.size
        num_detected_events = 0
        metrics = {
            'num_detected_events': 0,
            'detected_events': [0]*20,
            'event_support_sizes': [0]*20,
            'event_triggered_support_ratios': [0]*20,
            'event_triggered_support_sizes': [0]*20,
            'total_triggered_points': np.sum(triggered_labels.ravel()).item(),
            'total_triggered_ratio':(np.sum(triggered_labels.ravel())/ D).item(),
            'total_event_points': 0,
            'total_detected_support_ratio': 0.0,
            'total_detected_support_size': 0
        }
        for i in range(20):
            event_mask = self._get_event_mask(i)
            metrics['event_support_sizes'][i] = np.sum(event_mask).item()

            total_mask |= event_mask
            event_overlap_mask = event_mask * triggered_labels
            if np.any(event_overlap_mask):
                metrics['detected_events'][i]= 1
                metrics['num_detected_events'] += 1
            metrics['event_triggered_support_sizes'][i] = np.sum(event_overlap_mask)
            metrics['event_triggered_support_ratios'][i] = ( metrics['event_triggered_support_sizes'][i] 
                                                                / metrics['event_support_sizes'][i])
        
        metrics['total_event_points'] = np.sum(total_mask).item()
        metrics['total_detected_support_size'] = np.sum(triggered_labels[total_mask]).item()
        metrics['total_detected_support_ratio'] = (metrics['total_detected_support_size'] 
                                                    / metrics['total_event_points'])
        metrics['detected_triggered_ratio'] = (metrics['total_detected_support_size'] 
                                                    / max([metrics['total_triggered_points'],1]))
        if keys is not None:
            return {key: metrics[key] for key in keys if key in metrics}
        else:
            return metrics
        

    def anomaly_scoring_score(self, anomaly_scores, ratios=RATIOS):
        indd = np.flip(np.argsort(anomaly_scores, axis=None))
        metrics = defaultdict(list)
        metrics['ratios'] = ratios

        for i, ratio in enumerate(ratios):
            ind = np.unravel_index(indd[:int(len(indd)*ratio)], anomaly_scores.shape)
            topk_event_idx = ind
            anomaly_mask = np.zeros(anomaly_scores.shape, dtype=bool)
            anomaly_mask[topk_event_idx] =1

            anomaly_detection_scores = self.anomaly_detection_score(anomaly_mask)
            for key, value in anomaly_detection_scores.items():
                metrics[key].append(value)
        metrics['total_detected_events'] = sum(metrics['num_detected_events'])
        return metrics
            


