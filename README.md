# scNiche v1.1.0

## Identification and characterization of cell niches in tissue from spatial omics data at single-cell resolution

[![python >=3.9](https://img.shields.io/badge/python-%3E%3D3.9-brightgreen)](https://www.python.org/) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14195486.svg)](https://doi.org/10.5281/zenodo.14195486)

scNiche is a computational framework to identify and characterize cell niches from single-cell spatial omics data

![avatar](images/workflow.jpg)

## Requirements and Installation
[![anndata 0.10.1](https://img.shields.io/badge/anndata-0.10.1-success)](https://pypi.org/project/anndata/) [![pandas 1.5.0](https://img.shields.io/badge/pandas-1.5.0-important)](https://pypi.org/project/pandas/) [![squidpy 1.2.3](https://img.shields.io/badge/squidpy-1.2.3-critical)](https://pypi.org/project/squidpy/) [![scanpy 1.9.1](https://img.shields.io/badge/scanpy-1.9.1-informational)](https://github.com/scverse/scanpy) [![dgl 1.1.0+cu113](https://img.shields.io/badge/dgl-1.1.0%2Bcu113-blueviolet)](https://www.dgl.ai/)  [![torch 1.21.1+cu113](https://img.shields.io/badge/torch-1.12.1%2Bcu113-%23808080)](https://pytorch.org/get-started/locally/) [![matplotlib 3.6.2](https://img.shields.io/badge/matplotlib-3.6.2-ff69b4)](https://pypi.org/project/matplotlib/) [![seaborn 0.13.0](https://img.shields.io/badge/seaborn-0.13.0-9cf)](https://pypi.org/project/seaborn/) 

### Create and activate conda environment with requirements installed.
For scNiche, the Python version need is over 3.9. If you have already installed a lower version of Python, consider installing Anaconda, and then you can create a new environment.
```
cd scNiche-main

conda env create -f scniche_dev.yaml -n scniche
conda activate scniche
```

### Install PyTorch and DGL
We developed scNiche in a CUDA 11.3 environment. Here is an example of installing PyTorch and DGL with CUDA11.3:
```
# install PyTorch
pip install torch==1.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

# install DGL
pip install dgl==1.1.0+cu113 -f https://data.dgl.ai/wheels/cu113/repo.html
```
The version of PyTorch and DGL should be suitable to the CUDA version of your machine. You can find the appropriate version on the [PyTorch](https://pytorch.org/get-started/locally/) and [DGL](https://www.dgl.ai/) website.


### Install scNiche
```
python setup.py build
python setup.py install
```

## Tutorials (identify cell niches)
#### - Spatial proteomics data or single-cell spatial transcriptomics data

By default, scNiche requires the single-cell spatial omics data (stored as `.h5ad` format) as input, where cell population label of each cell needs to be provided. 

Here are examples of scNiche on simulated and biological datasets:
* [Demonstration of scNiche on the simulated data](tutorial/tutorial_simulated.ipynb)
* [Demonstration of scNiche on the mouse V1 neocortex STARmap data](tutorial/tutorial_STARmap.ipynb)


scNiche also provides a subgraph-based batch training strategy to scale to large datasets and multi-slices:

1. Batch training strategy of scNiche for single-slice:
* [Demonstration of scNiche on the mouse spleen CODEX data](tutorial/tutorial_spleen.ipynb) (over 80,000 cells per slice)

2. Batch training strategy of scNiche for multi-slices:
* [Demonstration of scNiche on the human upper tract urothelial carcinoma (UTUC) IMC data](tutorial/tutorial_utuc.ipynb) (containing 115,060 cells from 16 slices)
* [Demonstration of scNiche on the mouse frontal cortex and striatum MERFISH data](tutorial/tutorial_MERFISH.ipynb) (containing 376,107 cells from 31 slices)


#### - Low-resolution spatial transcriptomics data 
We here take 4 slices from the same donor of the [human DLPFC 10X Visium data](http://spatial.libd.org/spatialLIBD/) as an example.

In contrast to spatial proteomics data, which usually contain only a few dozen proteins, these spatial transcriptomics data can often measure tens of thousands of genes, 
with potential batch effects commonly present across tissue slices from different samples. 
Therefore, dimensionality reduction and batch effect removal need to be performed on the molecular profiles of the cells and their neighborhoods before run scNiche.
We used [scVI](https://github.com/scverse/scvi-tools) by defalut, however, simple PCA dimensionality reduction or other deep learning-based integration methods like [scArches](https://github.com/theislab/scarches) are also applicable.

Furthermore, cell type labels are usually unavailable for these spatial transcriptomics data. As alternatives, 
we can: 
1. Use the `deconvolution results of spots` as a substitute view to replace the `cellular compositions of neighborhoods`. 
We used the human middle temporal gyrus (MTG) scRNA-seq data by [Hodge et al.](https://doi.org/10.1038/s41586-019-1506-7) as the single-cell reference, and deconvoluted the spots using [Cell2location](https://github.com/BayraktarLab/cell2location):

* [Demonstration of scNiche on Slice 151673 (with deconvolution results)](tutorial/tutorial_dlpfc151673.ipynb)

2. Only use the molecular profiles of cells and neighborhoods as input:

* [Demonstration of scNiche on Slice 151673 (without deconvolution results)](tutorial/tutorial_dlpfc151673-2view.ipynb)


Multi-slice analysis of 4 slices based on the batch training strategy of scNiche:

* [Demonstration of scNiche on 4 slices from the same donor (with deconvolution results)](tutorial/tutorial_DLPFC.ipynb)

#### - Spatial multi-omics data 
The strategy of scNiche for modeling features from different views of the cell offers more possible avenues for expansion, 
such as application to spatial multi-omics data. We here ran scNiche on a postnatal day (P)22 mouse brain coronal section 
dataset generated by [Zhang et al.](https://doi.org/10.1038/s41586-023-05795-1), which includes RNA-seq and CUT&Tag (acetylated histone H3 Lys27 (H3K27ac) histone modification) modalities.
The dataset can be downloaded [here](https://zenodo.org/records/10362607).

* [Demonstration of scNiche on the mouse brain spatial CUT&Tag–RNA-seq data](tutorial/tutorial_multi-omics.ipynb)


## Tutorials (characterize cell niches)
scNiche also offers a downstream analytical framework for characterizing cell niches more comprehensively.

Here are examples of scNiche on two biological datasets:
* [Demonstration of scNiche on the human triple-negative breast cancer (TNBC) MIBI-TOF data](tutorial/tutorial_tnbc.ipynb)
* [Demonstration of scNiche on the mouse liver Seq-Scope data](tutorial/tutorial_liver.ipynb)


## Acknowledgements
The scNiche model is developed based on the [multi-view clustering framework (CMGEC)](https://github.com/wangemm/CMGEC-TMM-2021). We thank the authors for releasing the codes.

## About
scNiche is developed by Jingyang Qian. Should you have any questions, please contact Jingyang Qian at qianjingyang@zju.edu.cn.

## References
Qian, J., Shao, X., Bao, H. et al. Identification and characterization of cell niches in tissue from spatial omics data at single-cell resolution. Nat Commun 16, 1693 (2025). [https://doi.org/10.1038/s41467-025-57029-9](https://doi.org/10.1038/s41467-025-57029-9)
