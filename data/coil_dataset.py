import os

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.neighbors import kneighbors_graph
from numpy.linalg import norm
# import gifsicle

from src.multilinear_ops.t2m import t2m

class CoilDataset:
    def __init__(self, angle_subsample_rate=2, img_res=128, batch=False, **kwargs):
        self.num_angles = 72//angle_subsample_rate
        self.img_res = img_res
        self.angle_subsample_rate = angle_subsample_rate
        self.dataset_path = kwargs.get('dataset_path', 
                                       os.path.join(os.getcwd(), '..', '..', 'data', 'coil-20-proc'))

        self.obj_ids = kwargs.get('obj_ids', list(range(1, 21)))
        if not batch:
            self.Ls = []
            self.Gs = []
            self.objects = [np.zeros([self.num_angles, self.img_res, self.img_res]) for _ in self.obj_ids]
            for i, obj_id in enumerate(self.obj_ids):
                obj_path = os.path.join(self.dataset_path, f'obj{obj_id}__')
                for a in range(0,72,angle_subsample_rate): # angles
                    image = Image.open(obj_path+f'{a}.png')
                    im = image.resize((self.img_res,self.img_res))
                    self.objects[i][a//angle_subsample_rate,:,:] = np.asarray(im)
                CD_Ang =  t2m(self.objects[i],1)
                CD_C = t2m(self.objects[i],2)
                CD_R = t2m(self.objects[i],3)
                A_C = kneighbors_graph(CD_C, 2, mode='connectivity', include_self=False)
                G_C = nx.from_numpy_array(A_C)
                A_R = kneighbors_graph(CD_R, 2, mode='connectivity', include_self=False)
                G_R = nx.from_numpy_array(A_R)
                A_Ang = kneighbors_graph(CD_Ang, 3, mode='distance',p=2,  include_self=False)
                G_Ang = nx.from_numpy_array(A_Ang)
                for _,_,d in G_Ang.edges(data=True):
                    d['weight'] = np.exp(-(d['weight']/(norm(G_Ang,2)*100))**2)
                Gs = [G_Ang, G_C, G_R]
                Ls = [nx.laplacian_matrix(g).toarray() for g in Gs]
                self.Ls.append(Ls)
                self.Gs.append(Gs)
        else:
            self.objects, self.Ls, self.Gs = get_coil(angle_subsample_rate, len(self.obj_ids), img_res)
    
    def plot_object(self, obj_id, angle=0, **kwargs):
        """Plot the object with the given ID and angle with k-NN graphs of the modes
        
        Args:
            obj_id (int): Object ID
            angle (int): Angle of the object to plot
            **kwargs: Additional arguments for plotting
                extra_images: List of extra images to plot
                extra_img_options: List of dictionaries with title and options for extra images
                figsize: Figure size
                supylabel: Super ylabel for the figure
                object: Object tensor
                Ls: List of Laplacians
                Gs: List of graphs
                draw_options: List of dictionaries with options for drawing the graphs

            Returns:
                fig, axs: Figure and Axes objects
            """
        extra_imgs = kwargs.get('extra_images', [])
        extra_plots = kwargs.get('extra_plots', [])
        extra_plot_titles = kwargs.get('extra_plot_titles', [f'Extra Plot-{i}' for i in range(len(extra_plots))])
        extra_plot_options = kwargs.get('extra_plot_options', [{'cmap':'gray'} for _ in extra_plots])
        extra_img_titles = kwargs.get('extra_image_titles', [f'Extra Image-{i}' for i in range(len(extra_imgs))])
        extra_img_options = kwargs.get('extra_image_options', [{'cmap':'gray'} for _ in extra_imgs])
        fig, axs = plt.subplots(1, 4+len(extra_imgs), figsize=kwargs.get('figsize', (20+5*len(extra_imgs), 5)))
        fig.tight_layout(pad=3.0)
        fig.supylabel(kwargs.get('supylabel', None))
        obj = kwargs.get('object', self.objects[obj_id])
        # Ls = kwargs.get('Ls', self.Ls[obj_id])
        Gs = kwargs.get('Gs', self.Gs[obj_id])
        axs[0].imshow(obj[angle, :, :], cmap='gray')
        axs[0].set_title(f'Object {obj_id} from angle {angle*self.angle_subsample_rate}')
        axs[0].axis('off')
        titles = ['View Angle', 'Row', 'Column']
        draw_options = kwargs.get('draw_options', {'with_labels': True})
        for i, G in enumerate(Gs):
            nx.draw_circular(G, ax=axs[i+1], **draw_options[i])
            axs[i+1].set_title(f'{titles[i]} graph')
        
        for j in range(len(extra_imgs)):
            axs[j+4].axis('off')
            axs[j+4].imshow(extra_imgs[j], **extra_img_options[j])
            axs[j+4].set_title(extra_img_titles[j])

        for j in range(len(extra_plots)):
            axs[j+4+len(extra_imgs)].axis('off')
            axs[j+4+len(extra_imgs)].imshow(extra_plots[j], **extra_plot_options[j])
            axs[j+4+len(extra_imgs)].set_title(extra_plot_titles[j])
        return fig, axs
    
    def plot_objects(self, **kwargs):
        """Plot all objects with the given ID and angle with k-NN graphs of the modes"""
        obj_ids = kwargs.get('obj_ids', self.obj_ids)
        # Ls = kwargs.get('Ls', [self.Ls[obj_id-1] for obj_id in obj_ids])
        Gs = kwargs.get('Gs', [self.Gs[obj_id-1] for obj_id in obj_ids])
        objects = kwargs.get('objects', [self.objects[obj_id-1] for obj_id in obj_ids])
        angle = kwargs.get('angle', 0)
        angles = kwargs.get('angles', [angle]*len(obj_ids))
        fig, axs = plt.subplots(len(obj_ids), 4, figsize=kwargs.get('figsize', (20, 5*len(obj_ids))))
        fig.tight_layout(pad=3.0)
        axs[0, 0].axis('off')
        titles = ['View Angle', 'Row', 'Column']
        draw_options = kwargs.get('draw_options', {'with_labels': True})
        for i, obj_id in enumerate(obj_ids):
            axs[i, 0].axis('off')
            axs[i,0].set_ylabel(kwargs.get('ylabel', None))
            axs[i, 0].imshow(objects[i][angles[i], :, :], cmap='gray')
            axs[i, 0].set_title(f'Object {obj_id} at angle {angles[i]*self.angle_subsample_rate}')
            for j, G in enumerate(Gs[obj_id]):
                nx.draw_circular(G, ax=axs[i, j+1], **draw_options[j])
                axs[i, j+1].axis('off')
                axs[i, j+1].set_title(f'{titles[j]} graph')
        return fig, axs
    

def learn_object_graphs(object_tensor, **kwargs):
    """Find k-NN graphs for each of the modes of a given tensor."""
    Gs = []
    Ls = []
    CD_Ang =  t2m(object_tensor,1)
    CD_C = t2m(object_tensor,2)
    CD_R = t2m(object_tensor,3)
    knn_options = kwargs.get('knn_options', [{'n_neighbors': 2, 'mode': 'connectivity', 'include_self': False},
                                             {'n_neighbors': 2, 'mode': 'connectivity', 'include_self': False},
                                             {'n_neighbors': 3, 'mode': 'distance', 'p': 2, 'include_self': False}])
    knn_scales = kwargs.get('knn_scales', [100, 100, 100])
    A_C = kneighbors_graph(CD_C, **knn_options[0])
    G_C = nx.from_numpy_array(A_C)
    A_R = kneighbors_graph(CD_R, **knn_options[1])
    G_R = nx.from_numpy_array(A_R)
    A_Ang = kneighbors_graph(CD_Ang, **knn_options[2])
    G_Ang = nx.from_numpy_array(A_Ang)
    for i in range(3):
        if 'weight' in G_Ang.edges(data=True):
            for u,v,d in G_Ang.edges(data=True):
                d['weight'] = np.exp(-(d['weight']/(norm(G_Ang,2)*100))**2)
    for _,_,d in G_Ang.edges(data=True):
        d['weight'] = np.exp(-(d['weight']/(norm(G_Ang,2)*knn_scales[i]))**2)
    Gs = [G_Ang, G_C, G_R]
    Ls = [nx.laplacian_matrix(g).toarray() for g in Gs]
    return Ls, Gs

def animate_object(object_tensor, path, **kwargs):
    """Animate the object and save as a GIF file"""
    fig, ax = plt.subplots()
    ax.axis('off')
    ims = []
    for i in range(object_tensor.shape[0]):
        im = ax.imshow(object_tensor[i, :, :], cmap='gray')
        ims.append([im])




def get_coil(comp=2, obj_ids=list(range(1,21)), res=128):
    """Imports the coil-20 dataset and returns the data in the form of a list of graphs and their laplacians

    Args:
        comp (int): Subsampling factor for different views
        Nobj (int): Number of objects to import
        res (int): Image resolution

    Returns:
        batched_coil: Tensor of order 4
        Ls: List of Laplacian matrices
        Gs: List of graphs
    """
    Nobj = len(obj_ids)
    batched_coil = np.zeros([Nobj,72//comp, res,res])
    cwd= os.getcwd()
    coil_path = os.path.join(cwd,'..','..','data','coil-20-proc')
    for o, id in enumerate(obj_ids): # objects
        obj_path = os.path.join(coil_path,f'obj{obj_ids}__')
        for a in range(0,72,comp): # angles
            image = Image.open(obj_path+f'{a}.png')
            im = image.resize((res,res))
            batched_coil[o,a//comp,:,:] = np.asarray(im)
    CD_Ang =  t2m(batched_coil,2)
    CD_C = t2m(batched_coil,3)
    CD_R = t2m(batched_coil,4)
    A_C = kneighbors_graph(CD_C, 2, mode='connectivity', include_self=False);
    G_C = nx.from_numpy_array(A_C); 
    A_R = kneighbors_graph(CD_R, 2, mode='connectivity', include_self=False);
    G_R = nx.from_numpy_array(A_R);
    A_Ang = kneighbors_graph(CD_Ang, 3, mode='distance',p=2,  include_self=False);
    G_Ang = nx.from_numpy_array(A_Ang);
    for u,v,d in G_Ang.edges(data=True):
        d['weight'] = np.exp(-(d['weight']/(norm(G_Ang,2)*100))**2)
    Gs = [G_Ang, G_C, G_R]
    Ls = [nx.laplacian_matrix(g).toarray() for g in Gs]
    return batched_coil, Ls, Gs

def get_coil_object(comp, obj_id, res):
    """Imports the coil-20 dataset and returns the data in the form of a list of graphs and their laplacians

    Args:
        comp (int): Subsampling factor for different views
        Nobj (int): Number of objects to import
        res (int): Image resolution

    Returns:
        batched_coil: Tensor of order 4
        Ls: List of Laplacian matrices
        Gs: List of graphs
    """
    Coil_Data = np.zeros([72//comp, res,res])
    cwd= os.getcwd()
    coil_path = os.path.join(cwd,'..','..','data','coil-20-proc')
    obj_path = os.path.join(coil_path,f'obj{obj_id}__')
    for a in range(0,72,comp): # angles
        image = Image.open(obj_path+f'{a}.png')
        im = image.resize((res,res))
        Coil_Data[a//comp,:,:] = np.asarray(im)

    CD_Ang =  t2m(Coil_Data,1)
    CD_C = t2m(Coil_Data,2)
    CD_R = t2m(Coil_Data,3)
    A_C = kneighbors_graph(CD_C, 2, mode='connectivity', include_self=False);
    G_C = nx.from_numpy_array(A_C); 
    A_R = kneighbors_graph(CD_R, 2, mode='connectivity', include_self=False);
    G_R = nx.from_numpy_array(A_R);
    A_Ang = kneighbors_graph(CD_Ang, 3, mode='distance',p=2,  include_self=False);
    G_Ang = nx.from_numpy_array(A_Ang);
    for u,v,d in G_Ang.edges(data=True):
        d['weight'] = np.exp(-(d['weight']/(norm(G_Ang,2)*100))**2)
    Gs = [G_Ang, G_C, G_R]
    Ls = [nx.laplacian_matrix(g).toarray() for g in Gs]
    return Coil_Data, Ls, Gs

