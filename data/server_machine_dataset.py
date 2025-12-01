import os, sys
from pathlib import Path

DATA_DIR = Path(os.path.join(os.path.dirname(__file__)))
BASE_DIR = DATA_DIR.parent
SMD_DIR = DATA_DIR / 'server_machine_dataset'
SMD_TRAIN_DIR = SMD_DIR / 'train'
SMD_TEST_DIR = SMD_DIR / 'test'
SMD_TEST_LABEL_DIR = SMD_DIR / 'test_label'
SMD_TEST_INTERPRETATION_LABEL_DIR = SMD_DIR / 'interpretation_label'
sys.path.append(BASE_DIR.as_posix())

import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import networkx as nx
import torch
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, precision_recall_curve, auc

from src.multilinear_ops.matricize import matricize
from src.multilinear_ops.tensorize import tensorize

machine_num_channels = {
    1: 8, # Machine id: Number of channels
    2: 9,
    3: 11,
}

class SMDMachineChannel:
    """Represents a single channel of the server machine dataset."""
    
    def __init__(self, machine_id, channel_id, day_folding=(24,60), verbose=False,
                    center_data=False, mean_modes=[1],**kwargs):
        if machine_id not in machine_num_channels.keys():
            raise ValueError(f"Invalid machine ID: {machine_id}. Valid IDs are {list(machine_num_channels.keys())}.")
        if channel_id < 1 or channel_id > machine_num_channels[machine_id]:
            raise ValueError(f"Invalid channel ID: {channel_id}. Valid IDs for machine {machine_id} are 1 to {machine_num_channels[machine_id]}.")
        
        self.machine_id = machine_id
        self.channel_id = channel_id
        self.test_data = np.array(pd.read_csv(SMD_TEST_DIR / f'machine-{machine_id}-{channel_id}.txt',
                                              header=None)).T
        self.feature_mean = np.mean(self.test_data, axis=1, keepdims=True)
        
        self.test_labels = np.array(pd.read_csv(SMD_TEST_LABEL_DIR / f'machine-{machine_id}-{channel_id}.txt',
                                                header=None), dtype=bool).T
        self.anomaly_groups = []  # List to hold anomaly group support information
        self.og_mask = None       # Mask for the original data, does not include the sythetic padding

        self.n_features = self.test_data.shape[0]
        self.n_timepoints = self.test_data.shape[1]
        full_D = self.n_timepoints // 1440
        incomplete_D = int(np.ceil((self.n_timepoints / 1440) - full_D )) + full_D
        Y_dim = (self.n_features, incomplete_D, *day_folding)
        label_dim = (1, incomplete_D, *day_folding)
        
        Y = matricize( np.zeros(Y_dim, dtype=np.float64), [1], [2,3,4])
        labels = matricize(np.zeros(label_dim, dtype=bool),[1], [2,3,4])
        og_mask = matricize(np.zeros(Y_dim, dtype=bool), [1], [2,3,4])
        
        self.load_anomaly_groups(verbose=verbose)

        Y[:,:self.n_timepoints]=self.test_data
        labels[:,:self.test_labels.shape[1]]=self.test_labels
        og_mask[:,:self.test_data.shape[1]]=True
        self.Y = tensorize(Y, Y_dim, [1], [2,3,4])
        self._calculate_mode_wise_means(full_D, day_folding, mean_modes=mean_modes)

        if center_data:
            self.Y -= self.mean

        self.labels = tensorize(labels, label_dim, [1], [2,3,4])
        self.og_mask = tensorize(og_mask, Y_dim, [1], [2,3,4])
        # Pad the data
        mean_Y = np.mean(self.mean, axis=1, keepdims=True)
        mean_Y = np.broadcast_to(mean_Y, (self.Y.shape[0], 1, *self.Y.shape[2:]))
        pad_mask = ~self.og_mask[:, full_D, ...]
        self.Y[:, full_D, ...][pad_mask] = mean_Y[:, 0, ...][pad_mask]
        
        self.G = nx.from_scipy_sparse_array(kneighbors_graph(self.test_data, 2,
                                                                metric='euclidean',
                                                                mode='connectivity',
                                                                include_self=False))
        self._calculate_edge_weights(**kwargs.get('edge_weighting', {}))
        self.Gt = nx.grid_graph((self.Y.shape[-1], ), periodic=False)
    
    def _calculate_edge_weights(self, method=None, **kwargs):
        if method is not None:
            nodelist = list(self.G.nodes())
            edgelist = list(self.G.edges())
            B = nx.incidence_matrix(self.G,
                                        nodelist=nodelist, edgelist=edgelist, oriented=False).T
            diffs = B @ self.test_data
            if method == 'rbf':
                c = kwargs.get('c', 0.05)
                sigma_0 = np.sum(self.test_data**2)
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
                self.G.edges[edge]['weight'] = weights[i]

    def _calculate_mode_wise_means(self, full_D, day_folding, mean_modes=[1]):
        """Calculate the mean for each mode of the tensor."""
        self.mode_wise_means = []
        for i in range(4):
            Ym = matricize(self.Y[:,:full_D,...], [i+1])
            mode_mean = np.mean(Ym, axis=1)
            self.mode_wise_means.append(mode_mean)
        Ym = matricize(self.Y[:,:full_D,...], mean_modes)
        mean = np.mean(Ym, axis=1, keepdims=True)
        mshape = tuple([self.Y.shape[i] if i+1 in mean_modes else 1 for i in range(4)])
        self.mean = tensorize(mean, mshape, mean_modes)


    def estimate_noise_variance(self, method='truncated_svd', rank_threshold=0.99):
        if method=='truncated_svd':
            estimated_variances = []
            for i in range(4):
                Ym = matricize(self.Y, [i+1])
                svals = np.linalg.svd(matricize(self.Y, [i+1]), compute_uv=False)
                total_energy = np.sqrt(np.sum(svals**2))
                cumulative_energy = np.sqrt(np.cumsum(svals**2))
                r = (np.sum(cumulative_energy < (rank_threshold * total_energy))+1)
                residual_energy = total_energy - cumulative_energy[r-1]
                m, n = Ym.shape[0], Ym.shape[1]
                estimated_variances.append(residual_energy/(m*n - r*(m+n-r)))
            estimated_variance = np.mean(estimated_variances)
        else:
            raise ValueError(f"Unknown method: {method}. Supported methods are 'truncated_svd'.")
        return estimated_variance.item()

    def load_anomaly_groups(self, verbose=False):
        """Load the anomaly group support for a specific machine and channel."""
        file = SMD_TEST_INTERPRETATION_LABEL_DIR / f'machine-{self.machine_id}-{self.channel_id}.txt'
        if not file.exists():
            raise FileNotFoundError(f"Anomaly group support file does not exist: {file}")
        # Initialize the anomaly groups and labels
        self.st_test_labels = np.zeros((self.n_features, self.n_timepoints), dtype=bool)
        with open(file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: # Skip empty lines
                    continue
                anomaly_group = {}
                range_part, features_part = line.split(":")
                start, end = map(int, range_part.split("-"))
                features = np.array(list(map(int, features_part.split(","))))
                anomaly_group['start'] = start
                anomaly_group['end'] = end
                anomaly_group['features'] = features
                self.anomaly_groups.append(anomaly_group)
                self.st_test_labels[features-1, start:end] = True
        self.t_test_labels = self.st_test_labels.any(axis=0, keepdims=True)
        # Check if the loaded labels match the original test labels
        self.matched = (self.t_test_labels == self.test_labels).all()
        self.timepoint_anomaly_ratio = np.sum(self.test_labels)/ self.test_labels.size
        self.anomaly_ratio = np.sum(self.st_test_labels) / self.st_test_labels.size
        if verbose:
            print(f"Loaded {len(self.anomaly_groups)} anomaly groups for machine {self.machine_id}, channel {self.channel_id}.")
            print(f"Anomalous timepoints: {self.timepoint_anomaly_ratio*100:.3f}%")
            print(f"Anomalous features: {self.anomaly_ratio*100:.3f}%")
            print(f"Does interpretation labels cover the timepoint labels? {matched}")


    def anomaly_scoring_score(self, scores):
        """Return anomaly scoring metrics based on score tensor"""
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()
        
        if scores.shape == self.Y.shape:
            scores = matricize(scores, [1], [2,3,4])
        
        st_scores = scores[:, :self.n_timepoints]
        t_scores = st_scores.sum(axis=0)
        
        p, r, th = precision_recall_curve(self.test_labels.ravel(), t_scores.ravel())
        au_prc = auc(r, p)
        au_roc = roc_auc_score(self.test_labels.ravel(), t_scores.ravel())
        return {'au_prc': au_prc,
                'au_roc': au_roc,
                'st_scores': st_scores,
                't_scores': t_scores}

    def anomaly_detection_score(self, labels):
        """Return anomaly detection metrics based on labels tensor"""
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        
        labels = matricize(labels, [1], [2,3,4])
        st_labels = labels[:, :self.n_timepoints]
        t_labels = st_labels.any(axis=0)
        point_adjusted_labels = t_labels.copy()

        detected_group = 0
        group_support = 0
        group_detected_support = 0
        for i, ag in enumerate(self.anomaly_groups):
            features = ag['features'] - 1  # Convert to zero-based index
            start = ag['start']
            end = ag['end']
            if t_labels[start:end].any():
                detected_group += 1
                group_detected_support += np.sum(st_labels[features, start:end])
                group_support += len(features)*(end - start)
                point_adjusted_labels[start:end] = True
        
        raw_f1 = f1_score(self.test_labels.ravel(), t_labels.ravel(), zero_division=0)
        point_adjusted_f1 = f1_score(self.test_labels.ravel(), point_adjusted_labels.ravel(), zero_division=0)

        precision = precision_score(self.test_labels.ravel(), t_labels.ravel(), zero_division=0)
        point_adjusted_precision = precision_score(self.test_labels.ravel(), point_adjusted_labels.ravel(), zero_division=0)
        
        recall = recall_score(self.test_labels.ravel(), t_labels.ravel(), zero_division=0)
        
        point_adjusted_recall = recall_score(self.test_labels.ravel(), point_adjusted_labels.ravel(), zero_division=0)
        group_accuracy = detected_group / len(self.anomaly_groups)
        group_support_ratio = group_detected_support / group_support if group_support > 0 else 0
        return {
            'f1': raw_f1,
            'PA_F1': point_adjusted_f1,
            'precision': precision,
            'recall': recall,
            'PA_precision': point_adjusted_precision,
            'PA_recall': point_adjusted_recall,
            'group_accuracy': group_accuracy,
            'group_support_ratio': group_support_ratio
        }