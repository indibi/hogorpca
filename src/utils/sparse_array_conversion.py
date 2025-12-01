import scipy as sp
import torch


def convert_sp_sparse_to_sparse_torch(array):
    """Convert a SciPy sparse array to a PyTorch sparse tensor."""
    layout = array.format
    if layout == 'coo':
        return convert_sp_coo_to_torch_coo(array)
    else:
        return convert_sp_coo_to_torch_coo(array.tocoo())


def convert_sp_coo_to_torch_coo(sparse_coo_array):
    assert isinstance(sparse_coo_array, sp.sparse.coo_array)
    values = sparse_coo_array.data
    coordinates = sparse_coo_array.coords
    shape = sparse_coo_array.shape
    return torch.sparse_coo_tensor(torch.tensor(coordinates), torch.tensor(values), torch.Size(shape)).coalesce()
