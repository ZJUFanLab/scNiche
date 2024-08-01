import copy
from ._utils import *
from dgl.dataloading import GraphDataLoader


def prepare_data(
        adata: AnnData,
        k_cutoff_graph: int = 20,
        mik_graph: int = 5,
        verbose: bool = True
):

    feat1 = adata.obsm['X_cn_norm']
    feat2 = adata.obsm['X_data']
    feat3 = adata.obsm['X_data_nbr']

    if verbose:
        print("-------Constructing graph for each view...")
    for view, feat in zip(['g1', 'g2', 'g3'], [feat1, feat2, feat3]):
        g = construct_graph(np.array(feat), k_cutoff_graph, mik_graph)
        adata.uns[view] = g
    if verbose:
        print("Constructing done.")

    return adata


def prepare_data_batch(
        adata: AnnData,
        batch_num: int = 4,
        k_cutoff_graph: int = 20,
        mik_graph: int = 5,
        verbose: bool = True
):
    feat1 = adata.obsm['X_cn_norm']
    feat2 = adata.obsm['X_data']
    feat3 = adata.obsm['X_data_nbr']

    # TODO: batch idx
    random.seed(123)
    batch_size = adata.shape[0] // batch_num
    left_cell_num = adata.shape[0] % batch_num
    add_cell_num = batch_num - left_cell_num
    add_cell = random.choices(range(adata.shape[0]), k=add_cell_num)

    batch_idx = random_split(adata.shape[0], batch_size)
    if left_cell_num > 0:
        for i in range(left_cell_num):
            batch_idx[i].append(batch_idx[len(batch_idx) - 1][i])
        batch_idx = batch_idx[:-1]

        batch_idx_new = copy.deepcopy(batch_idx)
        for i in range(len(add_cell)):
            batch_idx_new[i + left_cell_num].append(add_cell[i])
    else:
        batch_idx_new = copy.deepcopy(batch_idx)

    adata.uns['batch_idx'] = batch_idx_new

    g1_list = []
    g2_list = []
    g3_list = []
    if verbose:
        print("-------Constructing batch-graph for each view...")
    for i in tqdm(range(batch_num)):
        feat1_tmp = feat1[batch_idx_new[i]]
        feat2_tmp = feat2[batch_idx_new[i]]
        feat3_tmp = feat3[batch_idx_new[i]]

        g1_tmp = construct_graph(np.array(feat1_tmp), k_cutoff_graph, mik_graph)
        g2_tmp = construct_graph(np.array(feat2_tmp), k_cutoff_graph, mik_graph)
        g3_tmp = construct_graph(np.array(feat3_tmp), k_cutoff_graph, mik_graph)

        g1_list.append(g1_tmp)
        g2_list.append(g2_tmp)
        g3_list.append(g3_tmp)

    if verbose:
        print("Constructing done.")

    mydataset = myDataset(g1_list, g2_list, g3_list)
    dataloader = GraphDataLoader(mydataset, batch_size=1, shuffle=False, pin_memory=True)
    adata.uns['dataloader'] = dataloader

    return adata
