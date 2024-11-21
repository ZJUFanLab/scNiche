import torch
import pandas as pd
import numpy as np
import scanpy as sc
from anndata import AnnData
from tqdm import tqdm
from sklearn.cluster import KMeans
from collections import defaultdict
import concurrent.futures
from copy import deepcopy
from sklearn.metrics import fowlkes_mallows_score, mean_absolute_percentage_error


def shuffling(x, latent, device):
    idx = torch.arange(0, x.shape[0]).to(device)
    idx2 = torch.randperm(idx.size(0)).to(device)
    idx_shuffling = idx[idx2].unsqueeze(1)
    idx_shuffling = idx_shuffling.repeat(1, latent)
    return torch.gather(x, 0, idx_shuffling)


# https://github.com/CSOgroup/cellcharter/blob/main/src/cellcharter/tl/_autok.py
def mirror_stability(n_clusters, stability):
    stability = [
        stability[i: i + len(n_clusters) - 1] for i in range(0, len(stability), len(n_clusters) - 1)
    ]
    stability = list(map(list, zip(*stability)))
    return np.array([stability[i] + stability[i - 1] for i in range(1, len(stability))])


def cluster_stability(
        adata: AnnData,
        n_clusters: tuple,
        use_rep: str = 'X_scniche',
        max_runs: int = 10,
        convergence_tol: float = 1e-2,
        similarity_function: callable = None):
    n_clusters = list(range(*(max(1, n_clusters[0] - 1), n_clusters[1] + 2)))
    X = adata.obsm[use_rep]
    random_state = 0
    labels = defaultdict(list)
    stability = []
    if similarity_function is None:
        similarity_function = fowlkes_mallows_score

    previous_stability = None
    for i in range(max_runs):
        new_labels = {}

        pbar = tqdm(n_clusters)
        for k in pbar:
            clustering = KMeans(n_clusters=k, random_state=i + random_state)
            new_labels[k] = clustering.fit_predict(X)
            pbar.set_description(f"Iteration {i + 1}/{max_runs}")

        if i > 0:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                pairs = [
                    (new_labels[k], labels[k + 1][i])
                    for i in range(len(list(labels.values())[0]))
                    for k in list(labels.keys())[:-1]
                ]
                stability.extend(list(executor.map(lambda x: similarity_function(*x), pairs)))

            if previous_stability is not None:
                stability_change = mean_absolute_percentage_error(
                    np.mean(mirror_stability(n_clusters, previous_stability), axis=1),
                    np.mean(mirror_stability(n_clusters, stability), axis=1),
                )
                if stability_change < convergence_tol:
                    for k, new_l in new_labels.items():
                        labels[k].append(new_l)

                    print(
                        f"Convergence with a change in stability of {stability_change} reached after {i + 1} iterations"
                    )
                    break

            previous_stability = deepcopy(stability)

        for k, new_l in new_labels.items():
            labels[k].append(new_l)

    if max_runs > 1:
        stability = mirror_stability(n_clusters, stability)
    else:
        stability = None

    # best_k
    if max_runs <= 1:
        raise ValueError("Cannot compute stability with max_runs <= 1")
    stability_mean = np.array([np.mean(stability[k]) for k in range(len(n_clusters[1:-1]))])
    best_idx = np.argmax(stability_mean)
    best_k = n_clusters[best_idx + 1]

    robustness_df = pd.melt(
        pd.DataFrame.from_dict({k: stability[i] for i, k in enumerate(n_clusters[1:-1])}, orient="columns"),
        var_name="K",
        value_name="Stability",
    )

    adata.uns['robustness_df'] = robustness_df
    adata.uns['best_k'] = best_k


def clustering(adata: AnnData,
               target_k: int,
               clustering_method: str = 'kmeans',
               resolution: float = 0.5,
               n_neighbor: int = 20,
               use_rep: str = 'X_scniche',
               add_key: str = 'scNiche',
               ):
    assert (clustering_method.lower() in ['kmeans', 'leiden']), 'clustering_method must be `kmeans` or `leiden`!'
    X = adata.obsm[use_rep]
    label = None

    if clustering_method.lower() == 'kmeans':
        print(f"Applying K-Means Clustering with {target_k} target cluster numbers...")
        label = KMeans(n_clusters=target_k, random_state=123).fit_predict(X)

    elif clustering_method.lower() == 'leiden':
        print(f"Applying Leiden Clustering with {resolution} resolution...")
        adata_tmp = sc.AnnData(X)
        sc.pp.neighbors(adata_tmp, n_neighbors=n_neighbor)
        sc.tl.leiden(adata_tmp, resolution=resolution)
        label = list(adata_tmp.obs['leiden'].values)

    order = sorted(list(set(label)))
    category = ['Niche' + str(i) for i in order]

    label_new = ['Niche' + str(i) for i in label]
    adata.obs[add_key] = label_new
    adata.obs[add_key] = pd.Categorical(adata.obs[add_key], categories=category, ordered=True)

    return adata

