import copy
from ._utils import *
from dgl.dataloading import GraphDataLoader


def prepare_data(
        adata: AnnData,
        choose_views: Optional[list] = None,
        k_cutoff_graph: int = 20,
        mik_graph: int = 5,
        verbose: bool = True
):
    """
    Prepare training data by constructing graphs for each view.
    Args:
        adata: AnnData
            Anndata object.
        choose_views: Optional[list], defaults to None.
            A list of views to construct graphs for. If None, defaults to ['X_cn_norm', 'X_data', 'X_data_nbr'].
        k_cutoff_graph: int, defaults to 20.
            The number of nearest neighbors to consider when constructing the graph.
        mik_graph: int, defaults to 5.
            The minimum number of nearest neighbors identified for each sample in the multi-view mutual information maximization (MMIM) module.
        verbose: bool, defaults to True.
            Whether to print progress messages.
    Returns:
        adata: AnnData
            The updated Anndata object with constructed graphs stored in `adata.uns`.
    """
    if verbose:
        print("-------Constructing graph for each view...")
    if choose_views is None:
        choose_views = ['X_cn_norm', 'X_data', 'X_data_nbr']
    else:
        missing_views = [view for view in choose_views if view not in adata.obsm.keys()]
        if missing_views:
            raise ValueError(f"The following views are missing in adata.obsm: {', '.join(missing_views)}")

    for view in choose_views:
        feat = adata.obsm[view]
        g = construct_graph(np.array(feat), k_cutoff_graph, mik_graph)
        graph_name = 'g_' + view
        adata.uns[graph_name] = g
    if verbose:
        print("Constructing done.")

    return adata


def prepare_data_batch(
        adata: AnnData,
        choose_views: Optional[list] = None,
        batch_num: int = 4,
        k_cutoff_graph: int = 20,
        mik_graph: int = 5,
        verbose: bool = True
):
    """
    Prepare training data by constructing graphs for each view for batch training strategy.
    Args:
        adata: AnnData
            Anndata object.
        choose_views: Optional[list], defaults to None.
            A list of views to construct graphs for. If None, defaults to ['X_cn_norm', 'X_data', 'X_data_nbr'].
        batch_num: int, defaults to 4.
            The number of batches (subgraphs) to split for batch training.
        k_cutoff_graph: int, defaults to 20.
            The number of nearest neighbors to consider when constructing the graph.
        mik_graph: int, defaults to 5.
            The minimum number of nearest neighbors identified for each sample in the multi-view mutual information maximization (MMIM) module.
        verbose: bool, defaults to True.
            Whether to print progress messages.
    Returns:
        adata: AnnData
            The updated Anndata object with constructed graphs stored in `adata.uns['g_*']` and dataloader in `adata.uns['dataloader']`.
    """
    # create batch idx
    random.seed(123)
    batch_size = adata.shape[0] // batch_num
    left_cell_num = adata.shape[0] % batch_num
    add_cell_num = batch_num - left_cell_num
    add_cell = random.choices(range(adata.shape[0]), k=add_cell_num)

    # bug fixed
    if left_cell_num < batch_size:
        batch_idx = random_split(adata.shape[0], batch_size)
    else:
        batch_idx = random_split2(adata.shape[0], batch_num)

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

    # check
    if choose_views is None:
        choose_views = ['X_cn_norm', 'X_data', 'X_data_nbr']
    else:
        missing_views = [view for view in choose_views if view not in adata.obsm.keys()]
        if missing_views:
            raise ValueError(f"The following views are missing in adata.obsm: {', '.join(missing_views)}")

    feat = [adata.obsm[view] for view in choose_views]
    g_list = [[] for _ in range(len(feat))]

    if verbose:
        print("-------Constructing batch-graph for each view...")

    for i in tqdm(range(batch_num)):
        for j in range(len(feat)):
            feat_tmp = feat[j][batch_idx_new[i]]
            g_tmp = construct_graph(np.array(feat_tmp), k_cutoff_graph, mik_graph)
            g_list[j].append(g_tmp)

    if verbose:
        print("Constructing done.")

    mydataset = myDataset(g_list)
    dataloader = GraphDataLoader(mydataset, batch_size=1, shuffle=False, pin_memory=True)
    adata.uns['dataloader'] = dataloader

    return adata
