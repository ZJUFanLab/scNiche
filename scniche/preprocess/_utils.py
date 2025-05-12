import random
import numpy as np
import pandas as pd
import torch
import dgl
import scanpy as sc
from anndata import AnnData
from tqdm import tqdm
from sklearn.decomposition import PCA
from scipy.sparse import issparse
from torch.utils.data import Dataset, DataLoader
from sklearn.neighbors import NearestNeighbors
from typing import Optional, Union


def convert_adj(sparse_mat):
    """
    Convert `scipy.sparse.matrix` to `torch.sparse.tensor`
    """
    sparse_mat = sparse_mat.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mat.row, sparse_mat.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mat.data)
    shape = torch.Size(sparse_mat.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def cal_spatial_neighbors(
        adata: AnnData,
        celltype_key: str = 'Cell_type',
        celltype_order: Optional[list] = None,
        mode: str = 'KNN',
        rad_cutoff: Optional[float] = 20,
        k_cutoff: Optional[float] = 20,
        verbose: bool = True):

    assert (mode.lower() in ['radius', 'knn']), 'mode must be `radius` or `knn`!'
    if verbose:
        print("-------Calculating Cellular Neighborhoods...")

    # CNs
    meta = adata.obs.copy()
    meta['x_new'] = list(adata.obsm['spatial'][:, 0])
    meta['y_new'] = list(adata.obsm['spatial'][:, 1])

    if celltype_order is None:
        celltype_order = sorted(meta[celltype_key].unique())

    meta[celltype_key] = pd.Categorical(meta[celltype_key], categories=celltype_order, ordered=True)
    meta = pd.concat([meta, pd.get_dummies(meta[celltype_key])], axis=1)
    celltype = celltype_order
    values = meta[celltype].values

    if mode.lower() == 'radius':
        if verbose:
            print("Identifying neighbours within " + str(rad_cutoff) + " pixels of every cell...")
        nbr = NearestNeighbors(radius=rad_cutoff).fit(meta[['x_new', 'y_new']])
        _, indices = nbr.radius_neighbors(meta[['x_new', 'y_new']])
        cn = list()
        for i in range(len(indices)):
            cn_tmp = list(values[indices[i]].sum(axis=0))
            cn.append(cn_tmp)

    elif mode.lower() == 'knn':
        if verbose:
            print("Identifying the " + str(k_cutoff) + " nearest neighbours for every cell...")
        nbr = NearestNeighbors(n_neighbors=k_cutoff + 1).fit(meta[['x_new', 'y_new']])
        _, indices = nbr.kneighbors(meta[['x_new', 'y_new']])
        chunk = np.arange(len(indices))  # indices
        cn = values[indices.flatten()].reshape(len(chunk), (k_cutoff + 1), len(celltype)).sum(axis=1)

    cn = pd.DataFrame(cn)
    cn.index = meta.index
    cn.columns = celltype
    cn_norm = cn.div(cn.sum(axis=1), axis='rows')
    adata.uns['CN_order'] = celltype
    adata.obsm['X_cn'] = np.array(cn)
    adata.obsm['X_cn_norm'] = np.array(cn_norm)
    if verbose:
        print("Calculating done.")

    return adata


def cal_spatial_exp(
        adata: AnnData,
        layer_key: Optional[str] = None,
        is_pca: bool = False,
        n_comps: int = 50,
        mode: str = 'KNN',
        rad_cutoff: Optional[float] = 20,
        k_cutoff: Optional[float] = 20,
        verbose: bool = True):

    assert (mode.lower() in ['radius', 'knn']), 'mode must be `radius` or `knn`!'
    # adata.layers['data'] = adata.X.copy()
    if verbose:
        print("-------Calculating Neighboring expression...")

    coord = pd.DataFrame({'x': adata.obsm['spatial'][:, 0], 'y': adata.obsm['spatial'][:, 1]})

    if mode.lower() == 'radius':
        nbr = NearestNeighbors(radius=rad_cutoff).fit(coord)
        _, indices = nbr.radius_neighbors(coord)
    elif mode.lower() == 'knn':
        nbr = NearestNeighbors(n_neighbors=k_cutoff + 1).fit(coord)
        _, indices = nbr.kneighbors(coord)
        indices = np.delete(indices, 0, axis=1)

    # cell * gene
    if layer_key is not None:
        data_raw = adata.obsm[layer_key].copy()
    else:
        if issparse(adata.X):
            data_raw = adata.X.toarray().copy()
        else:
            data_raw = adata.X.copy()
    data_nbr = []
    for i in range(indices.shape[0]):
        data_nbr_tmp = data_raw[indices[i]].mean(axis=0)
        data_nbr.append(data_nbr_tmp)
    data_nbr = np.array(data_nbr)

    if is_pca:
        data_raw = PCA(n_components=n_comps).fit_transform(data_raw)
        data_nbr = PCA(n_components=n_comps).fit_transform(data_nbr)

    adata.obsm['X_data'] = data_raw
    adata.obsm['X_data_nbr'] = data_nbr

    if verbose:
        print("Calculating done.")

    return adata


def process_multi_slices(
        adata: AnnData,
        celltype_key: str = 'Cell_type',
        sample_key: str = 'Sample',
        mode: str = 'KNN',
        rad_cutoff: Optional[float] = 20,
        k_cutoff: Optional[float] = 20,
        layer_key: Optional[str] = None,
        is_pca: bool = False,
        n_comps: int = 50,
        verbose: bool = True):

    celltype_order = sorted(list(set(adata.obs[celltype_key])))
    sample_list = sorted(list(set(adata.obs[sample_key])))

    print(f"-------Process {len(sample_list)} slices...")
    adata_list = {}
    pbar = tqdm(sample_list)
    for i in pbar:
        adata_tmp = adata[adata.obs[sample_key] == i, ].copy()
        adata_list[i] = adata_tmp.copy()
        adata_list[i] = cal_spatial_neighbors(
            adata=adata_list[i], celltype_key=celltype_key, celltype_order=celltype_order, mode=mode,
            k_cutoff=k_cutoff, rad_cutoff=rad_cutoff, verbose=verbose
        )
        adata_list[i] = cal_spatial_exp(
            adata=adata_list[i], layer_key=layer_key, is_pca=is_pca, n_comps=n_comps, mode=mode,
            k_cutoff=k_cutoff, rad_cutoff=rad_cutoff, verbose=verbose
        )
        pbar.set_description('Sample: {}'.format(i))

    adata_new = sc.concat([adata_list[x] for x in sample_list], keys=None)
    adata_new.obsm['X_cn'] = np.nan_to_num(adata_new.obsm['X_cn'])
    adata_new.obsm['X_cn_norm'] = np.nan_to_num(adata_new.obsm['X_cn_norm'])
    adata_new.uns['CN_order'] = celltype_order

    return adata_new


def construct_graph(
        data: np.ndarray,
        knn: int = 10,
        mik: int = 5):

    # knn
    train_neighbors = NearestNeighbors(n_neighbors=knn + 1, metric='euclidean').fit(data)
    _, idx = train_neighbors.kneighbors(data)

    # mik
    mi_idx = idx[:, :mik]

    # adj
    adj = train_neighbors.kneighbors_graph(data)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    g = dgl.from_scipy(adj)
    g.ndata['feat'] = torch.FloatTensor(data).to(torch.float32)
    g.ndata['adj'] = convert_adj(adj)
    g.ndata['mik'] = torch.Tensor(mi_idx).type(torch.long)

    return g


# left_cell_num < batch_size
def random_split(n, m):
    nums = list(range(n))
    random.shuffle(nums)
    return [nums[i:i + m] for i in range(0, n, m)]


# left_cell_num > batch_size
def random_split2(n, batch_num):
    nums = list(range(n))
    random.shuffle(nums)

    batch_size = n // (batch_num + 1)
    result = [nums[i * batch_size: (i + 1) * batch_size] for i in range(batch_num)]
    result.append(nums[batch_num * batch_size:])

    return result


def set_seed():
    # seed
    seed = 123
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    dgl.random.seed(seed)


class myDataset(Dataset):
    def __init__(self, g_list):
        self.g_list = g_list

    def __getitem__(self, idx):

        return tuple(g[idx] for g in self.g_list)

    def __len__(self):
        return len(self.g_list[0])



