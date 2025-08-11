from anndata import AnnData
from typing import Optional
from scipy.stats import mannwhitneyu
from tqdm import tqdm
import pandas as pd
import numpy as np
import squidpy as sq
from concurrent.futures import ThreadPoolExecutor


def enrichment(adata: AnnData, id_key: str, val_key: str, library_key: Optional[str] = None):
    """
    Enrichment analysis for single-cell level data.
    For each cluster (defined by id_key), calculate the enrichment in specific groups (val_key) compared to other groups.
    Args:
        adata: AnnData
            Anndata object.
        id_key: str
            Key in `adata.obs` that defining clusters need to be enriched (e.g., `cell_type`).
        val_key: str
            Key in `adata.obs` that defining groups (e.g., `scNiche`).
        library_key: Optional[str], defaults to None.
            Key in `adata.obs` that defining samples (e.g., `library`). If None, creates dummy library (all cells in one sample).
            If the key already exists, it will be suffixed with '_new' to avoid conflicts.
    Returns:
        Enrichment results stored in `adata.uns`:
            - `{val_key}_{id_key}_fc`: DataFrame with fold changes of enrichment.
            - `{val_key}_{id_key}_pval`: DataFrame with p-values of enrichment from Mann-Whitney U test (one-sided, greater).
            - `{val_key}_{id_key}_proportion`: List of DataFrames with proportions of each group in each library.
    """
    print(f"Calculating the enrichment of each cluster ({id_key}) in group ({val_key})...")
    obs = adata.obs.copy()
    id_list = sorted(list(set(obs[id_key])))
    val_list = sorted(list(set(obs[val_key])))
    if library_key is None:
        library_key = 'library'
        if library_key in obs.columns:
            library_key = library_key + '_new'
        obs[library_key] = 1
    library_list = sorted(list(set(obs[library_key])))

    df = obs.groupby([library_key, id_key, val_key]).size().unstack().fillna(0)
    df_list = []
    for i in library_list:
        df_tmp = df.loc[(i,)]
        # avoid false positive (for example, (2 / 4 = 0.5) > (100 / 400 = 0.25))
        # min niche size
        MIN_NUM = 20
        df_tmp.loc[:, df_tmp.sum() < MIN_NUM] = 0

        df_list_tmp = df_tmp.div(df_tmp.sum(axis=0), axis=1)
        df_list.append(df_list_tmp)

    fc = []
    pval = []
    for idx in id_list:
        fc_tmp = []
        pval_tmp = []

        pbar = tqdm(val_list)
        for val in pbar:
            # prop
            observed = [df.loc[idx, val] for df in df_list]
            expected = [df.drop(val, axis=1).loc[idx, ].mean() for df in df_list]

            # filter NA, some niches don't exist in every library
            observed_new = [x for x in observed if not np.isnan(x)]
            expected_new = [x for x in expected if not np.isnan(x)]

            # calculate enrichment
            _, p_value = mannwhitneyu(observed_new, expected_new, alternative='greater')
            # add  a small value (1e-6) to avoind inf / -inf
            observed_mean = np.mean(observed_new)
            expected_mean = np.mean(expected_new)
            if observed_mean == 0:
                observed_mean = observed_mean + 1e-6
            if expected_mean == 0:
                expected_mean = expected_mean + 1e-6

            fold_change = np.log2(observed_mean / expected_mean)
            pval_tmp.append(p_value)
            fc_tmp.append(fold_change)

            pbar.set_description(f"Cluster: {idx}")

        pval.append(pval_tmp)
        fc.append(fc_tmp)

    pval = pd.DataFrame(pval)
    fc = pd.DataFrame(fc)
    pval.columns = fc.columns = val_list
    pval.index = fc.index = id_list

    adata.uns[f"{val_key}_{id_key}_fc"] = fc
    adata.uns[f"{val_key}_{id_key}_pval"] = pval
    adata.uns[f"{val_key}_{id_key}_proportion"] = df_list


def enrichment_spot(adata: AnnData, deconvoluted_res: pd.DataFrame, val_key: str, library_key: Optional[str] = None):
    """
    Enrichment analysis for spot level data.
    For each deconvoluted cluster (stored in `deconvoluted_res`), calculate the enrichment in specific groups (val_key) compared to other groups.
    Args:
        adata: AnnData
            Anndata object.
        deconvoluted_res: pd.DataFrame
            DataFrame with deconvoluted results, where columns are deconvoluted clusters and rows are spots.
        val_key: str
            Key in `adata.obs` that defining groups (e.g., `scNiche`).
        library_key: Optional[str], defaults to None.
            Key in `adata.obs` that defining samples (e.g., `library`). If None, creates dummy library (all cells in one sample).
            If the key already exists, it will be suffixed with '_new' to avoid conflicts.
    Returns:
        Enrichment results stored in `adata.uns`:
            - `{val_key}_{id_key}_fc`: DataFrame with fold changes of enrichment.
            - `{val_key}_{id_key}_pval`: DataFrame with p-values of enrichment from Mann-Whitney U test (one-sided, greater).
            - `{val_key}_{id_key}_proportion`: List of DataFrames with proportions of each group in each library.
    """
    print(f"Calculating the enrichment of each deconvoluted cluster in group ({val_key})...")
    id_key = 'celltype'
    obs = adata.obs.copy()

    # normalize
    deconvoluted_res = deconvoluted_res.div(deconvoluted_res.sum(axis=1), axis='rows')

    id_list = sorted(list(set(deconvoluted_res.columns)))
    val_list = sorted(list(set(obs[val_key])))
    if library_key is None:
        library_key = 'library'
        if library_key in obs.columns:
            library_key = library_key + '_new'
        obs[library_key] = 1
    library_list = sorted(list(set(obs[library_key])))

    choose_cols = obs[[library_key, val_key]]
    combined_df = deconvoluted_res.join(choose_cols)
    grouped = combined_df.groupby([library_key, val_key]).sum()
    stacked = grouped.stack().reset_index(name='value')
    stacked.rename(columns={'level_2': id_key}, inplace=True)

    df = (
        stacked.pivot_table(
            index=[library_key, id_key],
            columns=val_key,
            values='value',
            fill_value=0
        )
    )

    # df = obs.groupby([library_key, id_key, val_key]).size().unstack().fillna(0)
    df_list = []
    for i in library_list:
        df_tmp = df.loc[(i,)]
        # avoid false positive (for example, (2 / 4 = 0.5) > (100 / 400 = 0.25))
        # min niche size
        MIN_NUM = 20
        df_tmp.loc[:, df_tmp.sum() < MIN_NUM] = 0

        df_list_tmp = df_tmp.div(df_tmp.sum(axis=0), axis=1)
        df_list.append(df_list_tmp)

    fc = []
    pval = []
    for idx in id_list:
        fc_tmp = []
        pval_tmp = []

        pbar = tqdm(val_list)
        for val in pbar:
            # prop
            observed = [df.loc[idx, val] for df in df_list]
            expected = [df.drop(val, axis=1).loc[idx, ].mean() for df in df_list]

            # filter NA, some niches don't exist in every library
            observed_new = [x for x in observed if not np.isnan(x)]
            expected_new = [x for x in expected if not np.isnan(x)]

            # calculate enrichment
            _, p_value = mannwhitneyu(observed_new, expected_new, alternative='greater')
            # add  a small value (1e-6) to avoind inf / -inf
            observed_mean = np.mean(observed_new)
            expected_mean = np.mean(expected_new)
            if observed_mean == 0:
                observed_mean = observed_mean + 1e-6
            if expected_mean == 0:
                expected_mean = expected_mean + 1e-6

            fold_change = np.log2(observed_mean / expected_mean)
            pval_tmp.append(p_value)
            fc_tmp.append(fold_change)

            pbar.set_description(f"Cluster: {idx}")

        pval.append(pval_tmp)
        fc.append(fc_tmp)

    pval = pd.DataFrame(pval)
    fc = pd.DataFrame(fc)
    pval.columns = fc.columns = val_list
    pval.index = fc.index = id_list

    adata.uns[f"{val_key}_{id_key}_fc"] = fc
    adata.uns[f"{val_key}_{id_key}_pval"] = pval
    adata.uns[f"{val_key}_{id_key}_proportion"] = df_list


def _remove_intra_cluster_links(
        adata: AnnData,
        cluster_key: str,
        connectivity_key: str = 'spatial_connectivities',
        distances_key: str = 'spatial_distances',
        copy: bool = False):
    """
    Remove intra-cluster links from spatial connectivity and distance matrices.
    Args:
        adata: AnnData
            Anndata object.
        cluster_key: str
            Key in `adata.obs` that defining clusters (e.g., `scNiche`).
        connectivity_key: str, defaults to 'spatial_connectivities'.
            Key in `adata.obsp` that defining the spatial connectivity matrix.
        distances_key: str, defaults to 'spatial_distances'.
            Key in `adata.obsp` that defining the spatial distance matrix.
        copy: bool, defaults to False.
            Whether to return copies of the modified matrices. If True, returns copies; if False, modifies in place.
    Returns:
        If copy is True, returns the modified connectivity and distance matrices.
    """
    conns = adata.obsp[connectivity_key].copy() if copy else adata.obsp[connectivity_key]
    dists = adata.obsp[distances_key].copy() if copy else adata.obsp[distances_key]

    for matrix in [conns, dists]:
        target_clusters = np.array(adata.obs[cluster_key][matrix.indices])
        source_clusters = np.array(
            adata.obs[cluster_key][np.repeat(np.arange(matrix.indptr.shape[0] - 1), np.diff(matrix.indptr))]
        )

        inter_cluster_mask = (source_clusters != target_clusters).astype(int)

        matrix.data *= inter_cluster_mask
        matrix.eliminate_zeros()

    if copy:
        return conns, dists


def _observed_n_clusters_links(adj, labels):
    """
    Calculate the observed number of spatial links between clusters.
    Args:
        adj: np.ndarray
            Spatial connectivity matrix.
        labels: Series
            Series of cluster labels.
    Returns:
        DataFrame with observed number of links between clusters.
    """
    labels_unique = labels.cat.categories
    obs = np.zeros((len(labels_unique), len(labels_unique)))
    for i, l1 in enumerate(labels_unique):
        total_cluster_links = adj[labels == l1]

        for j, l2 in enumerate(labels_unique):
            other_cluster_links = total_cluster_links[:, labels == l2]

            obs[i, j] = np.sum(other_cluster_links)

    obs = pd.DataFrame(obs, columns=labels_unique, index=labels_unique)
    return obs


def spatial_link(adata: AnnData, cluster_key: str, only_inter: bool = True, normalize: bool = False,
                 connectivity_key: str = 'spatial_connectivities', distances_key: str = 'spatial_distances',):
    """
    Calculate the spatial connectivities between clusters (defined by cluster_key).
    Args:
        adata: AnnData
            Anndata object.
        cluster_key: str
            Key in `adata.obs` that defining clusters (e.g., `scNiche`).
        only_inter: bool, defaults to True.
            Whether to only consider inter-cluster links. If True, intra-cluster links will be removed.
        normalize: bool, defaults to False.
            Whether to normalize the number of links by the number of cells in the source cluster.
        connectivity_key: str, defaults to 'spatial_connectivities'.
            Key in `adata.obsp` that defining the spatial connectivity matrix.
        distances_key: str, defaults to 'spatial_distances'.
            Key in `adata.obsp` that defining the spatial distance matrix.
    Returns:
        DataFrame with spatial link results stored in `adata.uns[f"{cluster_key}_spatial_link"]`.
    """
    adata_select = adata.copy()

    # spatial graph
    sq.gr.spatial_neighbors(adata_select, delaunay=True)

    if only_inter:
        _remove_intra_cluster_links(adata_select, cluster_key=cluster_key,
                                    connectivity_key=connectivity_key, distances_key=distances_key)

    adj = adata_select.obsp[connectivity_key]
    label = adata_select.obs[cluster_key]
    observed = _observed_n_clusters_links(adj, label)

    if normalize:
        for i in list(set(label)):
            for j in list(set(label)):
                if observed.loc[i, j] == 0:
                    continue
                observed.loc[i, j] = observed.loc[i, j] / adata_select[adata_select.obs[cluster_key] == i].shape[0]

    adata.uns[f"{cluster_key}_spatial_link"] = observed


def _calculate_composition_ratio(df, library_key, niche_key, celltype_key, cutoff, selected):
    """
    Calculate the cell composition of each niche for a specific sample.
    Args:
        df: DataFrame
            DataFrame of observations with columns for library, niche, and cell type.
        library_key: str
            Key for the sample column in the DataFrame (e.g., `library`).
        niche_key: str
            Key for the niche column in the DataFrame (e.g., `scNiche`).
        celltype_key: str
            Key for the cell type column in the DataFrame (e.g., `cell_type`).
        cutoff: float
            The minimum ratio of niche to keep in the analysis.
        selected: str
            The specific sample to select for the analysis.
    Returns:
        DataFrame with cell composition results for each niche within each sample.
    """
    df_select = df[df[library_key] == selected]
    niche_ratios = df_select[niche_key].value_counts(normalize=True)
    niche_to_keep = niche_ratios[niche_ratios >= cutoff].index

    result = []
    for niche in niche_to_keep:
        df_slice = df_select[df_select[niche_key] == niche]
        celltype_ratios = df_slice[celltype_key].value_counts(normalize=True)
        niche_ratio = niche_ratios[niche]
        row = {
            library_key: selected,
            niche_key: niche,
            'Niche_ratio': niche_ratio,
            **{f'{c}_ratio': ratio for c, ratio in celltype_ratios.items()}
        }
        result.append(row)

    return pd.DataFrame(result)


def _calculate_average_exp(df, library_key, niche_key, celltype_key, gene_list, cutoff, selected_celltype, selected):
    """
    Calculate the average expression of genes of each cell type within each niche for a specific sample.
    Args:
        df: DataFrame
            DataFrame of observations with columns for library, niche, cell type, and gene expression.
        library_key: str
            Key for the sample column in the DataFrame (e.g., `library`).
        niche_key: str
            Key for the niche column in the DataFrame (e.g., `scNiche`).
        celltype_key: str
            Key for the cell type column in the DataFrame (e.g., `cell_type`).
        gene_list: list
            A list of specific genes to select for average expression calculation.
        cutoff: float
            The minimum ratio of niche to keep in the analysis.
        selected_celltype: list
            A list of specific cell types to select for average expression calculation.
        selected: str
            The specific sample to select for the analysis.
    Returns:
        DataFrame with average expression results of each cell type for each niche within each sample.
    """
    df_select = df[df[library_key] == selected]
    niche_ratios = df_select[niche_key].value_counts(normalize=True)
    niche_to_keep = niche_ratios[niche_ratios >= cutoff].index

    result = []
    for niche in niche_to_keep:
        df_slice = df_select[(df_select[niche_key] == niche) & (df_select[celltype_key].isin(selected_celltype))]
        avg_values = {gene: df_slice[gene].mean() for gene in gene_list}
        niche_ratio = niche_ratios[niche]
        row = {
            library_key: selected,
            niche_key: niche,
            'Niche_ratio': niche_ratio,
            **avg_values
        }
        result.append(row)

    return pd.DataFrame(result)


def calculate_composition_multi(adata: AnnData, library_key: str, niche_key: str, celltype_key: str, cutoff: float = 0.05):
    """
    Calculate the cell composition of each niche (defined by niche_key) across multi-samples (defined by library_key).
    Args:
        adata: AnnData
            Anndata object.
        library_key: str
            Key in `adata.obs` that defining libraries (e.g., `library`).
        niche_key: str
            Key in `adata.obs` that defining niches (e.g., `scNiche`).
        celltype_key: str
            Key in `adata.obs` that defining cell types (e.g., `cell_type`).
        cutoff: float, defaults to 0.05.
            The minimum ratio of niche to keep in the analysis.
    Returns:
        DataFrame with cell composition results stored in `adata.uns["composition_multi"]`.
    """
    obs = adata.obs.copy()
    obs[celltype_key] = obs[celltype_key].astype('str')

    with ThreadPoolExecutor() as executor:
        results = [
            executor.submit(
                _calculate_composition_ratio, obs, library_key, niche_key, celltype_key, cutoff, sample
            ) for sample in obs[library_key].unique()
        ]

        dfs = [r.result() for r in results]

    df = pd.concat(dfs, ignore_index=True)

    adata.uns["composition_multi"] = df


def calculate_average_exp_multi(adata: AnnData, layer_key: str, library_key: str, niche_key: str, celltype_key: str,
                                gene_list: Optional[list] = None, selected_celltype: Optional[list] = None, cutoff: float = 0.05):
    """
    Calculate the average expression of genes of each cell type within each niche (defined by niche_key) across multi-samples (defined by library_key).
    Args:
        adata: AnnData
            Anndata object.
        layer_key: str
            Key in `adata.layers` that defining the expression layer (e.g., `X`).
        library_key: str
            Key in `adata.obs` that defining libraries (e.g., `library`).
        niche_key: str
            Keyd in `adata.obs` that defining niches (e.g., `scNiche`).
        celltype_key: str
            Key in `adata.obs` that defining cell types (e.g., `cell_type`).
        gene_list: Optional[list], defaults to None.
            A list of specific genes to select. If None, all genes will be used.
        selected_celltype: Optional[list], defaults to None.
            A list of specific cell types to select. If None, all cell types will be used.
        cutoff: float, defaults to 0.05.
            The minimum ratio of niche to keep in the analysis.
    Returns:
        DataFrame with average expression results stored in `adata.uns["expression_multi"]`.
    """
    adata_use = adata.copy()
    if layer_key not in adata_use.layers.keys():
        adata_use.layers[layer_key] = adata_use.X
    exp = adata_use.to_df(layer=layer_key)
    if gene_list is None:
        gene_list = list(exp.columns)

    obs = adata_use.obs.copy()
    obs = pd.concat([obs, exp], axis=1)

    if selected_celltype is None:
        selected_celltype = list(set(obs[celltype_key]))

    with ThreadPoolExecutor() as executor:
        results = [
            executor.submit(
                _calculate_average_exp, obs, library_key, niche_key, celltype_key, gene_list, cutoff, selected_celltype, sample
            ) for sample in obs[library_key].unique()
        ]

        dfs = [r.result() for r in results]

    df = pd.concat(dfs, ignore_index=True)

    adata.uns["expression_multi"] = df


def average_exp(adata: AnnData, layer_key: str, id_key: str, val_key: str, select_idx: Optional[list] = None,
                select_val: Optional[list] = None):
    """
    Calculate the average expression of genes of for each cluster (defined by id_key) in each group (val_key).
    Args:
        adata: AnnData
            Anndata object.
        layer_key: str
            Key in `adata.layers` that defining the expression layer (e.g., `X`).
        id_key: str
            Key in `adata.obs` that defining clusters (e.g., `cell_type`).
        val_key: str
            Key in `adata.obs` that defining groups (e.g., `scNiche`).
        select_idx: Optional[list], defaults to None.
            A list of specific cluster indices to select. If None, all clusters will be used.
        select_val: Optional[list], defaults to None.
            A list of specific group values to select. If None, all groups will be used.
    Returns:
        DataFrame with average expression results.
    """
    adata_use = adata.copy()
    if select_idx is not None:
        adata_use = adata_use[adata_use.obs[id_key].isin(select_idx)].copy()

    if select_val is not None:
        adata_use = adata_use[adata_use.obs[val_key].isin(select_val)].copy()

    if layer_key not in adata_use.layers.keys():
        adata_use.layers[layer_key] = adata_use.X

    df = adata_use.to_df(layer=layer_key)
    average_df = df.groupby(adata_use.obs[id_key]).mean()

    return average_df


def cal_composition(adata: AnnData, id_key: str, val_key: str, ):
    """
    Calculate the proportion of each cluster (defined by id_key) in each groups (val_key).
    Args:
        adata: AnnData
            Anndata object.
        id_key: str
            Key in `adata.obs` that defining clusters (e.g., `cell_type`).
        val_key: str
            Key in `adata.obs` that defining groups (e.g., `scNiche`).
    Returns:
        DataFrame with proportion results.
    """
    obs = adata.obs.copy()
    df = obs.groupby([val_key, id_key]).size().unstack().fillna(0)
    df = df.div(df.sum(axis=1), axis=0)
    return df
