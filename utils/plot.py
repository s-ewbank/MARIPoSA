import scipy.io
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import rcParams, gridspec
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np
import pandas as pd
import networkx as nx
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
from matplotlib.patches import Patch, Ellipse, Rectangle
from utils import analyze
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA

def is_nonnum(value):
    try:
        int(value)
        return False
    except (ValueError, TypeError):
        return True

def fig_to_array(fig):
    canvas = FigureCanvas(fig)
    canvas.draw()
    width, height = canvas.get_width_height()
    image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(height, width, 3)
    return image

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def make_and_plot_ellipse(mean,
                          cov,
                          color,
                          label=None):
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]) * 180 / np.pi
    ell = Ellipse(mean, width=2 * np.sqrt(eigenvalues[0]), height=2 * np.sqrt(eigenvalues[1]),
                   angle=angle, facecolor=color, alpha=0.25, label=label,edgecolor="none")
    plt.gca().add_patch(ell)

def plot_module_usage(config,
                      usage_feats,
                      figW=4,
                      figH=2,
                      style="bar_scatter",
                      cmap="jet",
                      legend_pos="outside",
                      BORIS_to_pose_mat=None,
                      legend=True,
                      long_legend=False,
                      alt_labels=None,
                      alt_xticks=None,
                      title=None,
                      plot_stats=False):
    """
    Plot module usage

    :param config: config
    :param usage_feats: output of analyze.get_module_usage
    :param figW: figure width (default: 4)
    :param figH: figure height (default: 2)
    :param style: plot style; "bar_scatter", "bar_error", "points", or "stacked"
    :param cmap: colormap (default: jet)
    :param legend_pos: legend position (default: outside)
    :param BORIS_to_pose_mat: BORIS_to_pose_mat (default: None)
    :param legend: boolean to include or not include legend (default: False)
    :param long_legend: long legend (default: False)
    :param alt_labels: alternative group label dictionary (default: None)
    :param alt_xticks: alternative xticklabels (list), for stacked plot style only (default: None)
    :param title: plot title (default: None)
    :param plot_stats: include stats on plot (default: False)
    :return: fig

    """
    if ((style!="bar_scatter") and (style!="bar_error") and (style!="stacked") and (style!="points")):
        raise ValueError(f'style must be one of "bar_scatter", "bar_error", "stacked", or "points", not {style}')

    if len(np.unique(usage_feats.group_labels)) == 1:
        data_subgrouped = False
    else:
        data_subgrouped = True

    if not data_subgrouped:
        usage_df = usage_feats.to_df()
        usage_df.drop("group", axis=1, inplace=True)

        modules = usage_feats.feat_names
        n_modules = len(modules)

        bar_heights = np.mean(usage_df, axis=0)
        bar_sems = np.std(usage_df, axis=0) / np.sqrt(usage_df.shape[1])

        fig, ax = plt.subplots(figsize=(figW, figH), dpi=100)
        cmap = plt.get_cmap(cmap)
        if ((style == "bar_scatter") or (style == "bar_error")):
            ax.bar(
                x=np.arange(0, n_modules, 1),
                height=bar_heights,
                width=0.8,
                alpha=0.5,
                color=cmap([0.1])
            )
            if style == "bar_scatter":
                for i in range(len(usage_df.index)):
                    ax.scatter(np.arange(0, n_modules, 1) + np.random.normal(0, 0.01, n_modules),
                               usage_df.iloc[i],
                               color="black",
                               s=0.5
                               )
            elif style == "bar_error":
                ax.errorbar(
                    x=np.arange(0, n_modules, 1),
                    y=bar_heights,
                    yerr=bar_sems,
                    linestyle="none", linewidth=0.6,
                    color="black", capsize=1, markeredgewidth=0.75
                )
        elif style == "points":
            ax.errorbar(
                x=np.arange(0, n_modules, 1),
                y=bar_heights,
                yerr=bar_sems,
                color=cmap([0.1]),
                linestyle="none",
                marker="o", markersize=2.5, linewidth=0.75,
                capsize=2, markeredgewidth=0.75
            )
        ax.set_xlabel(config["data_source"] + ' Pose Label')
        ax.set_ylabel('Frequency')
        ax.set_xticks(np.arange(0, n_modules, 1), labels=[i.split("module")[1] for i in modules])
        ax.tick_params(axis='x', rotation=90, labelsize=plt.rcParams['font.size'] * 0.5, pad=2)
        plt.tight_layout()

    else:
        # To get groupnames in order
        groupnames = list(usage_feats.group_dict.keys())
        n_groups = len(groupnames)

        usage_df = usage_feats.to_df()

        modules = usage_feats.feat_names
        n_modules = len(modules)

        bar_heights = np.zeros([n_groups, n_modules])
        bar_sems = np.zeros([n_groups, n_modules])

        for g, group in enumerate(groupnames):
            subgroup_usage = usage_df[usage_df["group"] == group].copy()
            subgroup_usage.drop("group", axis=1, inplace=True)
            bar_heights[g, :] = subgroup_usage.mean(axis=0)
            bar_sems[g, :] = subgroup_usage.std(axis=0) / np.sqrt(subgroup_usage.shape[0])
        fig, ax = plt.subplots(figsize=(figW, figH), dpi=100)
        scale = 1 / (n_groups + .7)
        cmap = plt.get_cmap(cmap)
        colors = [cmap([i]) for i in np.linspace(0, 1, n_groups)]
        if ((style == "bar_scatter") or (style == "bar_error")):
            if style == "bar_scatter":
                for g, group in enumerate(groupnames):
                    subgroup_usage = usage_df[usage_df["group"] == group].copy()
                    subgroup_usage.drop("group", axis=1, inplace=True)
                    for i in subgroup_usage.index:
                        ax.scatter(
                            np.arange(0 + scale * g, n_modules + scale * g, 1) + np.random.normal(0, 0.1 * scale,
                                                                                                  n_modules),
                            subgroup_usage.loc[i],
                            color="black",
                            s=0.5
                        )
                bar_alpha = 0.5
            elif style == "bar_error":
                for g in range(n_groups):
                    ax.errorbar(
                        x=np.arange(0 + scale * g, n_modules + scale * g, 1),
                        y=bar_heights[g],
                        yerr=bar_sems[g],
                        linestyle="none", linewidth=0.6,
                        color="black", capsize=0.3, markeredgewidth=0.75, alpha=0.8
                    )
                bar_alpha = 0.85
            for g in range(n_groups):
                if alt_labels is not None:
                    label_g = alt_labels[groupnames[g]]
                else:
                    label_g = groupnames[g]
                ax.bar(
                    x=np.arange(0 + scale * g, n_modules + scale * g, 1),
                    height=bar_heights[g],
                    width=scale,
                    alpha=bar_alpha,
                    color=colors[g],
                    label=label_g
                )
        elif style == "points":
            for g in range(n_groups):
                if alt_labels is not None:
                    label_g = alt_labels[groupnames[g]]
                else:
                    label_g = groupnames[g]
                ax.errorbar(
                    x=np.arange(0, n_modules, 1),
                    y=bar_heights[g],
                    yerr=bar_sems[g],
                    color=colors[g],
                    label=groupnames[g],
                    linestyle="none",
                    marker="o", markersize=4, linewidth=0.75,
                    capsize=2, markeredgewidth=0.75
                )
        if style != "stacked":
            if legend_pos == "inside":
                ax.legend()
            elif legend_pos == "outside":
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

            ax.set_xlabel(config["data_source"] + ' Pose Label')
            ax.set_ylabel('Frequency')
            all_num_modules = np.sum([isinstance(module, str) for module in modules]) == 0
            if ((all_num_modules) and (n_modules >= 20)):
                xticks = np.arange(0, n_modules, 5, dtype=int)
                ax.set_xticks(xticks)
                ax.set_xticklabels(modules[xticks])
            else:
                ax.set_xticks(np.arange(0, n_modules, 1))
                ax.set_xticklabels(modules)
            ax.tick_params(axis='x', rotation=90, labelsize=plt.rcParams['font.size'] * 0.2 * n_groups, pad=2)
            if plot_stats:
                ylim = plt.ylim()
                plt.ylim(ylim[0], ylim[1] * 1.2)
                stat_result = usage_feats.f_oneway()
                for m, module in enumerate(stat_result["module"]):
                    if stat_result[stat_result["module"] == module]["p_uncorr"].values[0] < 0.001:
                        plt.plot([m - 0.1, m + 0.7], [ylim[1] * 1.05] * 2, color="black", linewidth=0.5)
                        plt.text(m + 0.3, ylim[1] * 1.08, "***", ha="center")
                    elif stat_result[stat_result["module"] == module]["p_uncorr"].values[0] < 0.01:
                        plt.plot([m - 0.1, m + 0.7], [ylim[1] * 1.05] * 2, color="black", linewidth=0.5)
                        plt.text(m + 0.3, ylim[1] * 1.08, "**", ha="center")
                    elif stat_result[stat_result["module"] == module]["p_uncorr"].values[0] < 0.05:
                        plt.plot([m - 0.1, m + 0.7], [ylim[1] * 1.05] * 2, color="black", linewidth=0.5)
                        plt.text(m + 0.3, ylim[1] * 1.08, "*", ha="center")
            plt.tight_layout()
        else:
            any_nonint = False
            if BORIS_to_pose_mat is None:
                for m, module in enumerate(modules):
                    if any_nonint:
                        module = m
                    if m == 0:
                        bar_bottom = np.zeros(n_groups)
                    ax.bar(np.arange(0, n_groups, 1), bar_heights[:, modules.index(module)], bottom=bar_bottom, align='center',
                           width=0.99)
                    ax.spines['top'].set_visible(False)
                    bar_bottom += bar_heights[:, modules.index(module)]
                    ax.spines['right'].set_visible(False)
                    ax.spines['top'].set_visible(False)
            else:
                modules = [int(i.split("module")[1]) for i in modules]
                behaviors = list(BORIS_to_pose_mat.index)
                if "other" not in behaviors:
                    behaviors = behaviors + ["other"]
                n_behaviors = len(behaviors)
                leg_handles_col0 = []
                leg_handles_col1 = []
                leg_labels = [""] * n_behaviors
                bottom = True
                cm_list = ['magma', 'Greys', 'YlOrBr', 'Blues', 'Purples', 'copper', 'Reds', 'BuGn', 'bone', 'YlGn',
                           'YlGnBu', 'YlOrRd']
                modules_plotted=[] #To account for modules not seen in remapping
                for remap in range(n_behaviors):
                    sub_modules = [int(i[0][0]) for i in config["remappings"] if i[1] == behaviors[remap]]
                    if (behaviors[remap]=="other"):
                        print([mod for mod in modules if ((mod not in modules_plotted) and (mod not in sub_modules))])
                    cmap = plt.get_cmap(cm_list[remap])
                    colors = [cmap([0.4]), cmap([0.5])]
                    leg_handles_col0.extend([Patch(facecolor=colors[0], edgecolor='none')])
                    leg_handles_col1.extend([Patch(facecolor=colors[1], edgecolor='none')])
                    if long_legend:
                        leg_labels.append(behaviors[remap] + "   (" + str(len(sub_modules)) + " modules)")
                    else:
                        leg_labels.append(behaviors[remap])  # +" "+str(sub_modules))
                    if len(sub_modules) > 0:
                        for c in sub_modules:
                            if c in modules:
                                if bottom == True:
                                    bar_bottom = np.zeros(n_groups)
                                    bottom = False
                                ax.bar(np.arange(0, n_groups, 1), bar_heights[:, modules.index(c)], bottom=bar_bottom, align='center',
                                       width=0.97, color=colors[c % 2])
                                ax.spines['top'].set_visible(False)
                                bar_bottom += bar_heights[:, modules.index(c)]
                                ax.spines['right'].set_visible(False)
                                ax.spines['top'].set_visible(False)
                                modules_plotted.append(c)
            ax.set_ylabel("Proportion of Time \nSpent in Pose Module")
            ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
            ax.set_xlim([-0.5, n_groups - 0.5])
            ax.set_xticks(np.arange(n_groups))
            if alt_xticks is None:
                ax.set_xticklabels(groupnames)
            else:
                ax.set_xticklabels(alt_xticks)
            if title is not None:
                ax.set_title(title)
            if legend == True:
                if BORIS_to_pose_mat is None:
                    modules = [int(i.split("module")[1]) for i in modules]
                    if len(modules) > 10:
                        leg_modules = []
                        for col in range(10):
                            items = [i for i in modules if int(i % 10) == col]
                            if len(items) > 1:
                                leg_modules.append("Modules " + ', '.join(map(str, items)))
                            else:
                                leg_modules.append("Module " + str(items[0]))
                    else:
                        leg_modules = modules
                    plt.legend(leg_modules, bbox_to_anchor=(1.05, 1))
                else:
                    leg_handles = leg_handles_col0 + leg_handles_col1
                    ax.legend(handles=leg_handles,
                              labels=leg_labels,
                              ncol=2, handletextpad=0.5, handlelength=1.0, columnspacing=-0.5, bbox_to_anchor=(1.05, 1))
            plt.tight_layout()
    return fig


def plot_distance_results(module_feature_object,
                          distance_results,
                          figW=3,
                          figH=3,
                          cmap="Blues",
                          title=None):
    """
    Plot results from distance computation

    :param module_feature_object:
    :param distance_results:
    :param figW:
    :param figH:
    :param cmap:
    :param title:
    :return:
    """
    fig = plt.figure(figsize=(figW, figH))
    dst = {}
    for grp in sorted(np.unique(module_feature_object.group_labels)):
        dst[grp] = distance_results[(((distance_results["grp_i"] == 0) & (distance_results["grp_j"] == grp)) | (
                (distance_results["grp_i"] == grp) & (distance_results["grp_j"] == 0)))]["distance_ij"].values

    cmap = plt.get_cmap(cmap)
    bplot = plt.boxplot([dst[i] for i in sorted(np.unique(module_feature_object.group_labels))],
                        positions=sorted(np.unique(module_feature_object.group_labels)),
                        patch_artist=True,
                        medianprops={'color': cmap([0.7]), 'linewidth': 2},
                        boxprops={'color': cmap([0.7])},
                        whiskerprops={'color': cmap([0.7]), 'linewidth': 2},
                        capprops={'color': cmap([0.7]), 'linewidth': 2},
                        flierprops={'markeredgecolor': cmap([0.7]), 'marker': 'o'})

    for patch in bplot['boxes']:
        patch.set_facecolor(cmap([0.2]))

    reverse_grp_dict = {v: k for k, v in module_feature_object.group_dict.items()}
    plt.xticks(sorted(np.unique(module_feature_object.group_labels)),
               labels=[reverse_grp_dict[i] for i in sorted(np.unique(module_feature_object.group_labels))],
               rotation=90)
    if title is not None:
        plt.title(title)

    plt.ylabel("Distance")
    plt.tight_layout()
    return fig


def network_plot(config,
                 labels_df=None,
                 module_usage=None,
                 module_transitions=None,
                 cmap="bwr",
                 include_labels=True,
                 scaling=1,
                 tscale=6,
                 figW=2.8,
                 figH=2.5,
                 alt_labels=None):
    """
    Plot network comparison

    :param config:
    :param labels_df:
    :param module_usage:
    :param module_feature_object:
    :param cmap:
    :param include_labels:
    :param scaling:
    :param tscale:
    :param figW:
    :param figH:
    :param alt_labels:
    :return:
    """
    if labels_df is not None:
        print("Defaulting to using labels_df to get module_usage and module_transitions")
        print("Getting module usage")
        module_usage = analyze.get_module_usage(config, labels_df)
        print("Getting module transitions")
        module_transitions = analyze.get_module_transitions(config, labels_df)

    g1 = module_usage.label_counts[np.array(module_usage.group_labels) == 0, :]
    g2 = module_usage.label_counts[np.array(module_usage.group_labels) == 1, :]
    n_modules = g1.shape[1]
    usage_stats = np.zeros(n_modules)
    for module in range(g1.shape[1]):
        usage_stats[module] = stats.ttest_ind(g2[:, module], g1[:, module]).statistic
    transition_stats = stats.ttest_ind(
        np.array(module_transitions.transition_count_matrices)[np.array(module_usage.group_labels) == 1],
        np.array(module_transitions.transition_count_matrices)[np.array(module_usage.group_labels) == 0]).statistic
    for i in range(n_modules):
        transition_stats[i, i] = 0
    fig, ax = plt.subplots(figsize=(figW, figH))
    G = nx.from_numpy_array(transition_stats, parallel_edges=True)
    edges = G.edges()
    pos = nx.circular_layout(G)
    weights = [G[u][v]['weight'] for u, v in edges]
    colormap = plt.get_cmap(cmap)
    colors = []
    np.min(weights)
    np.max(weights)
    for w in range(len(weights)):
        if weights[w] >= 0:
            col_w = colormap([0.95])
        elif weights[w] < 0:
            col_w = colormap([0.05])
        else:
            col_w = colormap([0])
        colors.append(col_w)
    G = nx.from_numpy_array(np.abs(transition_stats) * 5000, parallel_edges=True)
    edges = G.edges()
    pos = nx.circular_layout(G)
    scaling = scaling / n_modules
    weights = [G[u][v]['weight'] * scaling * 5 for u, v in edges]
    weights = list(np.array(weights) / 4000)
    labels = {}
    plotx = nx.draw_circular(G,
                             node_color=usage_stats,
                             cmap=plt.get_cmap(cmap),
                             vmin=-3,
                             vmax=3,
                             node_size=np.absolute(usage_stats) * 1000 * scaling,
                             edge_color=colors,
                             edgecolors=None,
                             with_labels=True,  # Keep this as True to display node labels
                             labels=labels,
                             font_size=500 * scaling,
                             font_weight="bold",
                             font_color="white",
                             width=weights)
    if include_labels == True:
        for m in range(n_modules):
            x, y = pos[m]
            label = str(m)
            label_font_size = 2 + 30 * scaling * (abs(usage_stats[m]) + 1) / 2
            plt.text(x, y, label, color="white", fontsize=label_font_size, fontweight="bold", ha="center", va="center")

    t_min = -tscale
    t_max = tscale
    smap = plt.cm.ScalarMappable(cmap=plt.get_cmap(cmap), norm=plt.Normalize(vmin=t_min, vmax=t_max))
    smap.set_array([])
    cbar_ax = fig.add_axes([0.85, 0.1, 0.05, 0.8])
    if alt_labels == None:
        label1 = list(module_usage.group_dict.keys())[0]
        label2 = list(module_usage.group_dict.keys())[1]
    else:
        label1 = alt_labels[list(module_usage.group_dict.keys())[0]]
        label2 = alt_labels[list(module_usage.group_dict.keys())[1]]

    cbar_ax.text(0, t_max * 1.1, f"{label2}>{label1}", va="bottom", color=plt.get_cmap(cmap)([0.9]))
    cbar_ax.text(0, t_min * 1.1, f"{label2}<{label1}", va="top", color=plt.get_cmap(cmap)([0.1]))
    cbar = plt.colorbar(smap, cax=cbar_ax, shrink=0.5)
    cbar.set_label('t-score')
    plt.margins(x=0.4, y=0.4)
    plt.subplots_adjust(right=0.85)
    return fig


def module_usage_sandplot(config, module_usage, BORIS_to_pose_mat=None, title=None, long_legend=True, figW=7, figH=3, convolve=False, window=5):
    """
    new sandplot function

    :param config:
    :param module_usage:
    :param BORIS_to_pose_mat:
    :param long_legend:
    :param convolve:
    :param window:
    :return:
    """

    fig, ax = plt.subplots(figsize=(figW, figH), dpi=100)

    data = module_usage.usage_density(convolve=convolve, window=window)

    data_mean = np.mean(data, axis=0).T
    bottom = np.zeros(data_mean[:, 0].shape)
    xpts = np.array(data[0].columns / 60, dtype=float)
    if BORIS_to_pose_mat is None:
        for m in range(data_mean.shape[1]):
            y_i = np.array(data_mean[:, m], dtype=float)
            plt.fill_between(xpts, bottom, bottom + y_i)
            bottom = y_i + bottom
    else:
        modules = list(data[0].index)
        behaviors = list(BORIS_to_pose_mat.index)
        if "other" not in behaviors:
            behaviors = behaviors + ["other"]
        n_behaviors = len(behaviors)
        leg_handles_col0 = []
        leg_handles_col1 = []
        leg_labels = [""] * n_behaviors
        cm_list = ['magma', 'Greys', 'YlOrBr', 'Blues', 'Purples', 'copper', 'Reds', 'BuGn', 'bone', 'YlGn',
                   'YlGnBu', 'YlOrRd']
        modules_plotted = []  # To account for modules not seen in remapping
        for remap in range(n_behaviors):
            sub_modules = [int(i[0][0]) for i in config["remappings"] if i[1] == behaviors[remap]]
            # if (behaviors[remap]=="other"):
            #     print([mod for mod in modules if ((mod not in modules_plotted) and (mod not in sub_modules))])
            cmap = plt.get_cmap(cm_list[remap])
            colors = [cmap([0.4]), cmap([0.5])]
            leg_handles_col0.extend([Patch(facecolor=colors[0], edgecolor='none')])
            leg_handles_col1.extend([Patch(facecolor=colors[1], edgecolor='none')])
            if long_legend:
                leg_labels.append(behaviors[remap] + "   (" + str(len(sub_modules)) + " modules)")
            else:
                leg_labels.append(behaviors[remap])  # +" "+str(sub_modules))
            if len(sub_modules) > 0:
                for c in sub_modules:
                    if c in modules:
                        bottom = np.array(bottom, dtype=float)
                        y_i = np.array(data_mean[:, modules.index(c)], dtype=float)
                        plt.fill_between(xpts, bottom, bottom + y_i, color=colors[c % 2])
                        bottom = y_i + bottom
                        modules_plotted.append(c)

    plt.ylim([0, 1])
    plt.xlim([xpts[0], xpts[-1]])
    plt.xlabel("Time (m)")
    plt.ylabel('Moving Average Proportion \nof Time Spent in Pose')
    if title!=None:
        plt.title(title)
    plt.tight_layout()
    return fig

def plot_keypoint_travel(keypoint_feature, cmap="viridis", plottype="band", figW=6, figH=3):
    """
    Plots displacement of a keypoint either over time or in bins from dist_df (output of analyze.dist_df_subgroups)

    :param dist_df: dist_df output from analyze.dist_df_subgroups
    :param cmap: matplotlib colormap
    :param plottype: type of plot ("band", "errorbar", or "bar" if no timebins)
    :param figW: figure width
    :param figH: figure height
    :return:
    """
    groups=list(keypoint_feature.group_dict.keys())
    n_groups=len(groups)
    fig, ax = plt.subplots(figsize=(figW, figH), dpi=100)
    if len(keypoint_feature.feat_names) == 1:
        plottype = "bar"
    cmap = plt.get_cmap(cmap)
    colors = [cmap([i]) for i in np.linspace(0, 1, n_groups)]
    xticks = keypoint_feature.feat_names
    for g, group in enumerate(groups):
        group_slice = np.array(keypoint_feature.group_labels)==keypoint_feature.group_dict[group]
        group_data = keypoint_feature.keypoint_feature[group_slice,:]
        group_mean = group_data.mean(axis=0)
        group_sem = group_data.std(axis=0) / np.sqrt(group_data.shape[1])
        if plottype == "errorbar":
            ax.errorbar(xticks, group_mean, label=group, yerr=group_sem, marker="o", capsize=2, color=colors[g])
            ax.legend()
            ax.set_xlabel("Time (m)")
        elif plottype == "band":
            ax.plot(xticks, group_mean, label=group, marker="o", color=colors[g])
            ax.fill_between(xticks, group_mean - group_sem, group_mean + group_sem, alpha=0.2, color=colors[g],
                            edgecolor="none")
            ax.legend()
            ax.set_xlabel("Time (m)")
        elif plottype == "bar":
            xticks = np.arange(0, n_groups, 1)
            ax.bar(xticks[g], group_mean, color=colors[g], alpha=0.6)
            ax.errorbar(xticks[g], group_mean, label=group, yerr=group_sem, capsize=2, color="black",
                        linestyle="none")
            if g == len(groups) - 1:
                ax.set_xticks(xticks)
                ax.set_xticklabels(groups)
    ax.set_ylabel("Locomotion (pix)")
    return fig

def plot_embeddings(module_feature_object, embeddings_object, figW=3, figH=3, cmap="viridis",title=None,legend=False,draw_ellipse=True,alt_legend=None):
    """
    Plot embeddings

    :param module_feature_object: module feature object (ModuleUsage or ModuleTransitions) from analyze.get_module_{xx}
    :param embeddings_object: embeddings object (LDA or PCA) from analyze.embed
    :param figW: figure width
    :param figH: figure height
    :param cmap: matplotlib colormap
    :param title: title string, or None
    :param legend: True or False
    :return: fig

    """
    if module_feature_object.__class__.__name__=="ModuleUsage":
        X_tfm = embeddings_object.transform(module_feature_object.label_counts)
    elif module_feature_object.__class__.__name__=="ModuleTransitions":
        X_tfm = embeddings_object.transform(module_feature_object.transition_counts)
    elif module_feature_object.__class__.__name__=="KeypointFeature":
        X_tfm = embeddings_object.transform(module_feature_object.keypoint_feature)
    y=module_feature_object.group_labels
    cmap = plt.get_cmap(cmap)
    colors=[cmap([i]) for i in np.arange(0,len(np.unique(y)),1/(len(np.unique(y))-0.9))]
    fig = plt.figure(figsize=(figW,figH))
    if embeddings_object.__class__==LDA:
        explained_variance = embeddings_object.explained_variance_ratio_
        plt.xlabel(f'LD1 ({explained_variance[0]*100:.2f}% variance explained)')
        plt.ylabel(f'LD2 ({explained_variance[1]*100:.2f}% variance explained)')
    if embeddings_object.__class__==PCA:
        explained_variance = embeddings_object.explained_variance_ratio_
        plt.xlabel(f'PC1 ({explained_variance[0]*100:.2f}% variance explained)')
        plt.ylabel(f'PC2 ({explained_variance[1]*100:.2f}% variance explained)')
    if title is not None:
        plt.title(title)
    legend_elements = {}
    for g in np.unique(y):
        legend_elements[g]=0
    reverse_group_dict = {v: k for k, v in module_feature_object.group_dict.items()}
    for obs in range(X_tfm.shape[0]):
        if legend_elements[module_feature_object.group_labels[obs]]==0:
            if alt_legend is not None:
                leg = alt_legend[reverse_group_dict[module_feature_object.group_labels[obs]]]
            else:
                leg = reverse_group_dict[module_feature_object.group_labels[obs]]
            plt.scatter(X_tfm[obs,0],X_tfm[obs,1],color=colors[module_feature_object.group_labels[obs]],
                        label=leg)
            legend_elements[module_feature_object.group_labels[obs]]=1
        else:
            plt.scatter(X_tfm[obs,0],X_tfm[obs,1],color=colors[module_feature_object.group_labels[obs]])
    if draw_ellipse==True:
        if embeddings_object.__class__==LDA:
            for r in np.unique(y):
                emb = X_tfm[y==r,:]
                mean = np.mean(emb, axis=0)
                cov = np.cov(emb, rowvar=False)
                make_and_plot_ellipse(mean, cov, color=colors[r])
        else:
            print(f"Ellipse only drawn for embeddings_object of class LDA, not {embeddings_object.__class__}")
    if legend:
        plt.legend(bbox_to_anchor=[1.05,1])
    plt.tight_layout()
    return fig


def plot_distance_matrix(module_feature_object, dist_mat,cmap="Greens",figW=3,figH=3,alt_labels=None,title=None):
    """
    Plot distance matrix

    :param module_feature_object:
    :param dist_mat:
    :param cmap:
    :param figW:
    :param figH:
    :param alt_labels:
    :param title:
    :return:
    """
    fig = plt.figure(figsize=(figW,figH),dpi=100)
    plt.imshow(dist_mat,aspect="auto",cmap=cmap)
    reverse_grp_dict = {v: k for k, v in module_feature_object.group_dict.items()}
    if alt_labels is not None:
        reverse_grp_dict = {v: alt_labels[k] for k, v in module_feature_object.group_dict.items()}
    xticks=[]
    xticklabels=[]
    for group in sorted(np.unique(module_feature_object.group_labels)):
        xticks.append(group)
        xticklabels.append(reverse_grp_dict[group])
        if group!=0:
            plt.axvline(group-0.5,color="black",linewidth=0.5)
    plt.xticks(ticks=xticks,labels=xticklabels,rotation=90)
    arr = np.array(module_feature_object.group_labels)
    edges = np.where(arr[:-1] != arr[1:])[0]
    for edge in edges:
        plt.axhline(edge+0.5,color="black",linewidth=0.5)
    ylabels = [reverse_grp_dict[i] for i in sorted(np.unique(module_feature_object.group_labels))]
    plt.yticks(ticks=np.concatenate([np.zeros(1),edges+1]),labels=ylabels)
    plt.ylabel("Observations")
    plt.xlabel("Centroids")
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    return fig

def plot_distance_box(module_feature_object, dist_mat, cmap="Blues",figW=3,figH=3,alt_labels=None,title=None):
    """
    Plot distance boxplot

    :param module_feature_object:
    :param dist_mat:
    :param cmap:
    :param figW:
    :param figH:
    :param alt_labels:
    :param title:
    :return:
    """
    fig = plt.figure(figsize=(figW, figH),dpi=100)
    dst = {}
    for grp in sorted(np.unique(module_feature_object.group_labels)):
        dst[grp] = dist_mat[np.array(module_feature_object.group_labels)==grp,0]

    cmap = plt.get_cmap(cmap)
    bplot = plt.boxplot([dst[i] for i in sorted(np.unique(module_feature_object.group_labels))],
                        positions=sorted(np.unique(module_feature_object.group_labels)),
                        patch_artist=True,
                        medianprops={'color': cmap([0.7]), 'linewidth': 2},
                        boxprops={'color': cmap([0.7])},
                        whiskerprops={'color': cmap([0.7]), 'linewidth': 2},
                        capprops={'color': cmap([0.7]), 'linewidth': 2},
                        flierprops={'markeredgecolor': cmap([0.7]), 'marker': 'o'})

    for patch in bplot['boxes']:
        patch.set_facecolor(cmap([0.2]))

    reverse_grp_dict = {v: k for k, v in module_feature_object.group_dict.items()}
    if alt_labels:
        reverse_grp_dict = {v: alt_labels[k] for k, v in module_feature_object.group_dict.items()}
    plt.xticks(sorted(np.unique(module_feature_object.group_labels)),
               labels=[reverse_grp_dict[i] for i in sorted(np.unique(module_feature_object.group_labels))],
               rotation=90)
    if title is not None:
        plt.title(title)

    plt.ylabel(f"Distance to {reverse_grp_dict[0]} Centroid")
    plt.tight_layout()
    return fig


# def plot_dist_boxplot(dists, dist_from, figW=5, figH=3,alt_labels=None):
#     """
#     Plot distance boxplots
#
#     :param dists: dists returned from lda_result.get_mahalanobis_distance()
#     :param dist_from: a group from the dataset from which to calculate all the distances
#     :param figW:
#     :param figH:
#     :return:
#     """
#
#     centroid=np.sum(["CENTROID" in i for i in list(dists.keys())])>0
#
#     if centroid:
#         pairings = [i for i in list(dists.keys()) if dist_from+"-CENTROID" in i]
#         centroid_lab = " Centroid"
#     else:
#         pairings = [i for i in list(dists.keys()) if dist_from in i]
#         centroid_lab=""
#
#     pairings_ticks = [i.replace("____vs____", "").replace(dist_from, "").replace("-CENTROID", "") for i in pairings]
#     if alt_labels is not None:
#         for p,tick in enumerate(pairings_ticks):
#             if tick=="":
#                 pairings_ticks[p]=dist_from
#         pairings_ticks = [alt_labels[p] for p in pairings_ticks]
#
#     fig = plt.figure(figsize=(figW, figH))
#     xticks = []
#     for p, pair in enumerate(pairings):
#         xticks.append(p)
#         c1 = "#4198B5"
#         c2 = "#D8EBF1"
#
#         box = plt.boxplot(np.array(dists[pair]), positions=[p], widths=0.5,
#                           patch_artist=True,
#                           boxprops=dict(facecolor=c2, color=c1),
#                           whiskerprops=dict(color=c1),
#                           capprops=dict(color=c1),
#                           medianprops=dict(color=c1),
#                           flierprops=dict(markerfacecolor=c1, marker='o', markersize=4))
#
#         dist_from_name=dist_from
#         if alt_labels is not None:
#             dist_from_name=alt_labels[dist_from]
#
#         plt.ylabel('Mahalanoubis Distance\nfrom '+dist_from_name+centroid_lab)
#         plt.yticks([])
#
#     plt.xticks(ticks=xticks, labels=pairings_ticks, rotation=90)
#     plt.tight_layout()
#
#     return fig
#
# def plot_pc_weights(pca,cmap="PuOr"):
#     """
#     Plot PCA weights
#
#     :param pca: pca object from sklearn
#     :return: fig
#     """
#     components = pca.components_
#     fig = plt.figure(figsize=(4,2),dpi=100)
#     pc_labels=["PC"+str(i+1) for i in range(pca.components_.shape[0])]
#     plt.imshow(components,cmap=cmap,vmin=-1,vmax=1)
#     plt.title("Principle Component Weights")
#     plt.xticks(np.arange(0,pca.components_.shape[1],1))
#     plt.xlabel("Pose Modules")
#     plt.yticks([0,1],labels=pc_labels)
#     plt.colorbar(cmap=cmap)
#     plt.tight_layout()
#     return fig
#
def BORIS_to_pose_matrix_plot(config, boris_to_pose_output, figW=4, figH=2.5, cmap="Greens",outline_top_match=True):
    fig, ax = plt.subplots(figsize=(figW, figH), dpi=100)
    plt.imshow(boris_to_pose_output.to_numpy(dtype='float'),cmap=cmap,aspect="auto",interpolation="none")
    data = boris_to_pose_output.to_numpy(dtype='float')
    num_cols = data.shape[1]
    if outline_top_match==True:
        for col in range(num_cols):
            max_row = np.argmax(data[:, col])
            if np.sum(data[:, col]==data[max_row, col])==1:
                rect = Rectangle((col - 0.5, max_row - 0.5), 1, 1, edgecolor='purple', facecolor='none', linewidth=0.5)
                ax.add_patch(rect)
    if len(boris_to_pose_output.columns>=20):
        xticks=np.arange(0,len(boris_to_pose_output.columns),5,dtype=int)
        xticklabels=np.arange(0,len(boris_to_pose_output.columns),1,dtype=int)[xticks]
    else:
        xticks=range(len(boris_to_pose_output.columns))
        xticklabels=boris_to_pose_output.columns

    plt.xticks(xticks,labels=xticklabels)
    plt.yticks(range(len(boris_to_pose_output.index)), labels=boris_to_pose_output.index)
    plt.xlabel(config["data_source"]+" Pose Module")
    plt.ylabel("Manually Scored\nBehavior")
    plt.tick_params(axis='x', rotation=90,
                    labelsize=plt.rcParams['font.size'] * 0.7, pad=2)
    plt.tight_layout()
    return fig

