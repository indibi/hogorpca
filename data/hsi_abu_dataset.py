"""Class for Airport-Beach-Urban Hyperspectral image Anomaly Detection dataset.

References
----------
..  [1] Xudong Kang, Xiangping Zhang, Shutao Li, Kenli Li, Jun Li, Jon Atli 
        Benediktsson, "Hyperspectral anomaly detection wtih attribute and
        edge-preserving filters" IEEE Transactions on Geoscience and Remote
        Sensing, 2017.
..  [2] “Data Sets.” Xudong Kang’s Homepage,
        http://xudongkang.weebly.com/data-sets.html. Accessed 3 Mar. 2026.
"""

import os, sys
from pathlib import Path

DATA_DIR = Path(os.path.join(os.path.dirname(__file__)))
BASE_DIR = DATA_DIR.parent
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



class ABUHyperSpectral:
    images_in_category = {
    'airport': 4, # Machine id: Number of channels
    'beach': 4,
    'urban': 5,
    }
    variance_estimates ={
        'airport': [1305.95757, 1468.8169574639946, 804.6095439103689, 1737.8740883588584],
        'beach': [52.587915302337855, 46.68342152213969, 1308.5898669842757, 3.856922056065146e-05],
        'urban': [1648.5846771513782, 1246.0974466058387, 1362.2746142092365, 1316.143822068098, 1598.6390705065924],
    }
    scaled_var_ests={
        'airport': [0.000491, 0.000541, 0.000602, 0.000385],
        'beach': [0.000041, 0.000068, 0.000378, 0.001383],
        'urban': [0.000545, 0.000412, 0.000443, 0.000866, 0.000770]
    }
    def __init__(self, category:str, image_id:int):
        self.category = category
        self.image_id = image_id
        hsi_data = sp.io.loadmat(
            DATA_DIR / 'hsi_abu' / f"abu-{category}-{image_id}.mat"
            )
        self.Y = hsi_data['data']
        self.labels = hsi_data['map']


    def anomaly_scoring_score(self, scores):
        """Return anomaly scoring metrics based on score tensor"""
        # if scores.ndim ==3:
        #     scores = (scores**2).sum(dim=2).sqrt()
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()
        p, r, th = precision_recall_curve(self.labels.ravel(), scores.ravel())
        au_prc = auc(r, p)
        au_roc = roc_auc_score(self.labels.ravel(), scores.ravel())
        return {'au_prc': au_prc,
                'au_roc': au_roc,
                }


    def anomaly_detection_score(self, labels):
        """Return anomaly detection metrics based on labels tensor"""
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        f1 = f1_score(self.labels.ravel(), labels.ravel(), zero_division=0)
        precision = precision_score(self.labels.ravel(), labels.ravel(), zero_division=0)
        recall = recall_score(self.labels.ravel(), labels.ravel(), zero_division=0)
        return {
            'f1': f1,
            'precision': precision,
            'recall': recall,
        }

    def animate(self, Y=None, save_dir:str|None = None):
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)
        if Y is None:
            Y = self.Y
        ax = axes[0]
        axes[1].imshow(self.labels);
        vmin = np.min(Y)
        vmax = np.max(Y)
        ims = [[ax.imshow(Y[...,i],
                          vmin=vmin, vmax=vmax,
                          animated=True)] for i in range(Y.shape[2])]
        ani = animation.ArtistAnimation(fig, ims,
                                        interval=50, blit=True,
                                        repeat_delay=1000)
        if save_dir is not None:
            writer = animation.PillowWriter(
                fps=15,
                metadata=dict(artist='mmi'),
                bitrate=1000)
            ani.save((save_dir +
                      f'abu_hsi_{self.category}_{self.image_id}.gif'
                      ),
                      writer=writer)
        plt.close()
        return fig, axes, ani