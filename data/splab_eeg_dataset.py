import os
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
from scipy import io
from sklearn.neighbors import kneighbors_graph
import yaml
import matplotlib.pyplot as plt

SPLAB_EEG_DIR = Path(os.path.join(os.path.dirname(__file__), 'splab_eeg'))
EEG_DATA_DIR = SPLAB_EEG_DIR / 'EEG data'
ELECTRODE_POSITIONS_FILE = SPLAB_EEG_DIR / 'electrode_positions.yaml'
CH_NAMES_FILE = SPLAB_EEG_DIR / 'EEG_electrodes.xlsx'

class EEGDataset:
    """Organizes the EEG dataset into tensor format and provides a proximity graph for the electrodes.

    The dataset is organized into 3D tensors of shape (n_trials, n_channels, n_timepoints) for each subject.
    ch_names is a list of the channel names corresponding to the electrodes in the tensors n_channels dimension.
    Matrices such as the adjacency matrix and the Laplacian matrix should be created respectively using the
    channel names and the proximity graph.
    """
    def __init__(self, G=None):
        """Construct the EEG dataset and the electrode proximity graph.

        The proximity graph is a NetworkX graph object where the nodes are the electrodes and the edges
        represent the binary proximity between electrodes calculated using the 4-nearest neighbors algorithm
        on the electrode positions.

        Parameters
        ----------
        G : networkx.Graph, optional
            A precomputed proximity graph. If not provided, a new graph is created using the electrode positions.
        
        """
        electrode_positions = yaml.safe_load(ELECTRODE_POSITIONS_FILE.read_text())
        electrode_positions = pd.DataFrame.from_dict(electrode_positions).reset_index(names=['Electrode'])
        electrode_positions['x']= electrode_positions['Radius']*np.cos((electrode_positions['Angle']+90)*np.pi/180)
        electrode_positions['y']= electrode_positions['Radius']*np.sin((electrode_positions['Angle']+90)*np.pi/180)
        self.electrode_positions = electrode_positions

        
        ch_names = pd.read_excel(CH_NAMES_FILE, sheet_name='Sheet1')
        ch_names = ch_names['Channels'].values
        self.ch_names =  [str(ch)[1:-1].upper() for ch in ch_names]

        if G is None:
            self.G = self._electrode_proximity_graph()
        
        self._load_dataset()
        self.adjacency_matrix = nx.adjacency_matrix(self.G, nodelist=self.ch_names)
        self.laplacian_matrix = nx.laplacian_matrix(self.G, nodelist=self.ch_names)


    def _electrode_proximity_graph(self):
        """Create a 4-nearest neighbors graph from the electrode positions."""
        A = kneighbors_graph(self.electrode_positions[['x', 'y']], n_neighbors=4,
                              mode='connectivity', include_self=False)
        # A = radius_neighbors_graph(electrode_positions[['x', 'y']], radius=0.42,
        #                                mode='connectivity', include_self=False) + A

        G = nx.from_scipy_sparse_array(A)
        G = nx.relabel_nodes(G, {i: self.electrode_positions['Electrode'][i]
                                 for i in range(len(self.electrode_positions))})
        
        G.add_edges_from([('FPZ', 'AF3')])
        return G
    

    def _load_dataset(self):
        eeg_dataset = []
        for i in range(1, 21):
            subject = f'{i:03}_L4_Er.mat'
            subject_file = EEG_DATA_DIR / subject
            subject_data = io.loadmat(subject_file)['data_elec'][0]
            subject_data = np.stack(subject_data, axis=0)
            eeg_dataset.append(np.moveaxis(subject_data, [0, 1, 2], [1, 0, 2]))
        self.subjects = eeg_dataset
        # min_number_of_trials = min([subject.shape[0] for subject in eeg_dataset])
        # return np.stack([eeg_dataset[i][:min_number_of_trials,...] for i in range(len(eeg_dataset))], axis=0)
    
    def draw_electrode_graph(self, **kwargs):
        """Draw the electrode proximity graph."""
        ax = kwargs.get('ax', None)
        node_size = kwargs.get('node_size', 300)
        node_color = kwargs.get('node_color', 'lightblue')
        font_size = kwargs.get('font_size', 8)
        node_list = kwargs.get('node_list', self.ch_names)
        nx.draw_networkx(self.G, ax=ax,
                        pos={self.electrode_positions['Electrode'][i]: 
                             (self.electrode_positions['x'][i],
                              self.electrode_positions['y'][i])
                                for i in range(len(self.electrode_positions))
                            },
                nodelist=node_list, #title=title, #cmap=cmap,
                 with_labels=True, node_size=node_size, node_color=node_color, font_size=font_size)
    
    
    def subject_means(self):
        """Return mean of trials for each subject organized as a 3D Tensor of shape (Subject, Channel, Time)"""
        subject_mean = [np.mean(subject, axis=0, keepdims=False) for subject in self.subjects]
        return np.stack(subject_mean, axis=0)
    
    def __repr__(self):
        """Return a string representation of the EEGDataset."""
        n_trials = [subject.shape[0] for subject in self.subjects]
        return f"""EEG Dataset:\n============
Number of subjects: {len(self.subjects)}
Number of channels: {len(self.ch_names)}
Number of timepoints per trial: {self.subjects[0].shape[2]}
Number of trials per subject: Mean={np.mean(n_trials)}, Median={np.median(n_trials)}, Min={np.min(n_trials)}, Max={np.max(n_trials)}
"""
    
    def count_number_of_anomalies_per_dim(self, anomalies):
        axis = [d for d in range(len(anomalies.shape))]
        counts = []
        for i, d in enumerate(axis):
            counts.append(np.sum(anomalies, axis=tuple([k for k in axis if k != d]), keepdims=False))
        return counts

    # def plot_outlier_heatmap(self, anomalies):
