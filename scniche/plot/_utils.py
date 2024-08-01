import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
from anndata import AnnData
from typing import Optional
import seaborn as sns
from . import palettes


def _assign_color(value, color: list):
    color_dict = dict()
    for i in range(len(value)):
        color_dict[value[i]] = color[i]
    return color_dict


def _convert_pval_to_asterisks(pval):
    if pval <= 0.0001:
        return "****"
    elif pval <= 0.001:
        return "***"
    elif pval <= 0.01:
        return "**"
    elif pval <= 0.05:
        return "*"
    return ""


def _set_palette(length):

    if length <= 10:
        palette = palettes.default_10
    elif length <= 20:
        palette = palettes.default_20
    elif length <= 28:
        palette = palettes.default_28
    elif length <= 57:
        palette = palettes.default_57
    elif length <= len(palettes.default_102):  # 103 colors
        palette = palettes.default_102
    else:
        palette = ['grey' for _ in range(length)]
        print(
            'the obs value has more than 103 categories. Uniform '
            "'grey' color will be used for all categories."
        )

    return palette


def _melt_df(df: DataFrame, library_key: str, select_niche: Optional[list] = None, order: Optional[list] = None, ):
    if select_niche is not None:
        df = df[df['scNiche'].isin(select_niche)]
    if order is not None:
        df['scNiche'] = pd.Categorical(df['scNiche'], categories=order)
    df_melt = pd.melt(df, id_vars=[library_key, 'scNiche', 'Niche_ratio'])
    return df_melt


def stacked_barplot(adata: AnnData, x_axis: str, y_axis: str, mode: str = 'proportion', palette: Optional[list] = None,
                    save: bool = False, save_dir: str = '', kwargs: dict = {}):
    assert (mode.lower() in ['proportion', 'absolute']), 'mode should be either `proportion` or `absolute`!'

    length = len(adata.obs[y_axis].astype('category').cat.categories)
    if palette is None:
        palette = _set_palette(length=length)
    df = adata.obs.groupby([x_axis, y_axis]).size().unstack().fillna(0)
    if mode.lower() == 'proportion':
        df = df.div(df.sum(axis=1), axis=0)

    # plot
    ax = df.plot(kind='bar', stacked=True, width=0.75, color=palette, linewidth=0, **kwargs)
    ax.legend(bbox_to_anchor=(1, 0.5), loc='center left',
              ncol=(1 if length <= 14 else 2 if length <= 30 else 3), frameon=False)
    if save:
        plt.savefig(save_dir, format='svg')


def enrichment_heatmap(adata: AnnData, id_key: str, val_key: str, binarized: bool = False, show_pval: bool = False,
                       col_order: Optional[list] = None, row_order: Optional[list] = None, anno_key: Optional[str] = None,
                       anno_palette: Optional[list] = None, save: bool = False, save_dir: str = '', kwargs: dict = {}):

    obs = adata.obs.copy()

    # id_key: columns; val_key: row
    fc = adata.uns[f"{val_key}_{id_key}_fc"].T.copy()
    pval = adata.uns[f"{val_key}_{id_key}_pval"].T.copy()

    # pvalue, only enrichment
    pval[fc <= 0] = 1
    kwargs['vmin'] = 0

    # set order
    if col_order is not None:
        fc = fc[col_order]
        pval = pval[col_order]
        kwargs['col_cluster'] = False
    if row_order is not None:
        fc = fc.loc[row_order]
        pval = pval.loc[row_order]
        kwargs['row_cluster'] = False

    if anno_key is not None:
        df_sub = obs.drop_duplicates(subset=val_key)
        df_sub.index = df_sub[val_key]
        anno = df_sub[anno_key]
    else:
        anno = fc.index

    length = len(anno.unique())
    if anno_palette is None:
        anno_palette = _set_palette(length=length)
    row_colors = dict(zip(anno.unique(), anno_palette))

    # binarized
    if binarized:
        fc = fc.applymap(lambda x: 0 if x <= 0 else 1)
        kwargs['vmax'] = 1
        kwargs['cmap'] = ['white', 'green']
    else:
        if 'cmap' not in kwargs.keys() or kwargs['cmap'] is None:
            cmap = 'magma'
        else:
            cmap = kwargs['cmap']
        # min -> white
        cmap = sns.color_palette(cmap, as_cmap=True)
        cmap.set_under('white')
        kwargs['cmap'] = cmap

    # plot heatmap
    if show_pval:
        pval_str = pval.applymap(_convert_pval_to_asterisks)
        kwargs['annot'] = pval_str
        kwargs['fmt'] = ''

    ax = sns.clustermap(fc, method='complete', row_colors=anno.map(row_colors), **kwargs)

    # figure legend
    for label, color in row_colors.items():
        ax.ax_col_dendrogram.bar(0, 0, color=color, label=label, linewidth=0)
    ax.ax_col_dendrogram.legend(
        bbox_to_anchor=(1, 0.5), loc='center left',
        ncol=(1 if length <= 14 else 2 if length <= 30 else 3), frameon=False
    )

    # Here labels on the y-axis are rotated
    for tick in ax.ax_heatmap.get_yticklabels():
        tick.set_rotation(0)

    if save:
        plt.savefig(save_dir, format='svg')


def multi_lineplot(adata: AnnData, library_key: str, show_list: list, mode: str = 'composition', select_niche: Optional[list] = None,
                   order: Optional[list] = None, palette: Optional[list] = None,
                   save: bool = False, save_dir: str = '', kwargs: dict = {}):
    assert (mode.lower() in ['composition', 'expression']), 'mode should be either `composition` or `expression`!'

    if mode.lower() == 'composition':
        df = adata.uns['composition_multi'].copy()
        show_list = [str(ct) + '_ratio' for ct in show_list]
    elif mode.lower() == 'expression':
        df = adata.uns['expression_multi'].copy()

    df_melt = _melt_df(df=df, library_key=library_key, select_niche=select_niche, order=order)
    df_melt_plot = df_melt[df_melt['variable'].isin(show_list)]

    length = len(show_list)
    if palette is None:
        palette = _set_palette(length=length)

    ax = sns.lineplot(data=df_melt_plot, x="scNiche", y="value", marker='o', hue='variable', palette=palette, **kwargs)
    plt.xticks(rotation=90)
    ax.legend(bbox_to_anchor=(1, 0.5), loc='center left',
              ncol=(1 if length <= 14 else 2 if length <= 30 else 3), frameon=False)

    if save:
        plt.savefig(save_dir, format='svg')


def multi_boxplot(adata: AnnData, library_key: str, show_list: list, mode: str = 'composition', select_niche: Optional[list] = None,
                  order: Optional[list] = None, palette: Optional[list] = None, show_scatter: bool = True,
                  save: bool = False, save_dir: str = '', boxplot_kwargs: dict = {}, scatter_kwargs: dict = {}):
    assert (mode.lower() in ['composition', 'expression']), 'mode should be either `composition` or `expression`!'

    if mode.lower() == 'composition':
        df = adata.uns['composition_multi'].copy()
        show_list = [str(ct) + '_ratio' for ct in show_list]
    elif mode.lower() == 'expression':
        df = adata.uns['expression_multi'].copy()

    df_melt = _melt_df(df=df, library_key=library_key, select_niche=select_niche, order=order)
    df_melt_plot = df_melt[df_melt['variable'].isin(show_list)]

    length = len(show_list)
    if palette is None:
        palette = _set_palette(length=length)

    ax = sns.boxplot(data=df_melt_plot, x="variable", y="value", hue='scNiche', gap=0.2, showfliers=False,
                     palette=palette, **boxplot_kwargs)
    if show_scatter:
        ax = sns.stripplot(data=df_melt_plot, x="variable", y="value", hue='scNiche', dodge=True, jitter=True,
                           legend=False, palette=palette, **scatter_kwargs)
    plt.xticks(rotation=90)
    ax.legend(bbox_to_anchor=(1, 0.5), loc='center left',
              ncol=(1 if length <= 14 else 2 if length <= 30 else 3), frameon=False)

    if save:
        plt.savefig(save_dir, format='svg')


def multi_linrplot_group(adata: AnnData, library_key: str, show: str, group1: list, group2: list, group1_niche: Optional[list] = None,
                         group2_niche: Optional[list] = None, mode: str = 'composition', group_name_list: Optional[list] = None,
                         palette: Optional[list] = None, save: bool = False, save_dir: str = '', kwargs: dict = {}):
    assert (mode.lower() in ['composition', 'expression']), 'mode should be either `composition` or `expression`!'

    if mode.lower() == 'composition':
        df = adata.uns['composition_multi'].copy()
        show = str(show) + '_ratio'
    elif mode.lower() == 'expression':
        df = adata.uns['expression_multi'].copy()

    df_melt = _melt_df(df=df, library_key=library_key, select_niche=None, order=None)

    df_group1 = df_melt[(df_melt[library_key].isin(group1)) & (df_melt['scNiche'].isin(group1_niche))]
    df_group2 = df_melt[(df_melt[library_key].isin(group2)) & (df_melt['scNiche'].isin(group2_niche))]
    df_group1['scNiche'] = pd.Categorical(df_group1['scNiche'], categories=group1_niche)
    df_group2['scNiche'] = pd.Categorical(df_group2['scNiche'], categories=group2_niche)

    df_group1_plot = df_group1[df_group1['variable'] == show]
    df_group2_plot = df_group2[df_group2['variable'] == show]

    if palette is None:
        palette = _set_palette(length=2)

    fig, ax1 = plt.subplots()
    sns.lineplot(data=df_group1_plot, x='scNiche', y='value', ax=ax1, marker='o', color=palette[0], **kwargs)
    plt.xticks(rotation=90)
    ax2 = ax1.twiny()
    sns.lineplot(data=df_group2_plot, x='scNiche', y='value', ax=ax2, marker='s', color=palette[1], **kwargs)
    plt.xticks(rotation=90)

    if group_name_list is None:
        group_name_list = ['Group1', 'Group2']

    ax1.legend([group_name_list[0]], bbox_to_anchor=(1, 0.55), loc='center left', frameon=False)
    ax2.legend([group_name_list[1]], bbox_to_anchor=(1, 0.45), loc='center left', frameon=False)

    plt.title(show)

    if save:
        plt.savefig(save_dir, format='svg')









