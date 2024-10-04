import scipy.io
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import rcParams, gridspec
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import networkx as nx
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
from matplotlib.patches import Ellipse
from utils import analysis

def plot_module_usage(config,labels_df,start,stop,figW=4,figH=2,style="bar_scatter",cmap="jet"):
    """
    Plots the frequency of the pose modules occurring by group in the labels dataframe output by the label_counter function.

    :param labels_df: dataframe output from analysis.label_counter
    :param start: time in seconds to start dataframe from.
    :param stop: time in seconds to stop dataframe at.
    :param fps: frames per second of recording.
    :param figW: figure width
    :param figH: figure height
    :param style: "bar_scatter", "bar_error", or "points"
    :return:
    """
    fps = int(config["fps"])
    start_frame = start*fps
    stop_frame = stop*fps
    labels_df=labels_df[start_frame:stop_frame]
    total_frames = stop_frame-start_frame
    labels_flat = np.array(labels_df)
    labels_flat = [item for sublist in labels_flat for item in sublist]
    modules = np.unique(labels_flat)
    n_modules = len(modules)

    label_counts = np.zeros([labels_df.shape[1], n_modules])
    for i in range(labels_df.shape[1]):
        for m in range(n_modules):
            module = modules[m]
            try:
                module = np.int64(module)
            except ValueError:
                pass
            label_counts[i,m] = np.count_nonzero \
                (labels_df[[labels_df.columns[i]]] == module) /total_frames
    bar_heights = np.mean(label_counts,axis=0)
    bar_sems = np.std(label_counts,axis=0)/np.sqrt(labels_df.shape[1])

    fig, ax = plt.subplots(figsize=(figW,figH),dpi=100)
    cmap = plt.get_cmap(cmap)
    if ((style=="bar_scatter") or (style=="bar_error")):
        ax.bar(
            x=np.arange(0, n_modules, 1),
            height=bar_heights,
            width=0.8,
            alpha=0.5,
            color=cmap([0.1])
        )
        if style=="bar_scatter":
            for i in range(len(label_counts)):
                ax.scatter(np.arange(0, n_modules, 1) + np.random.normal(0, 0.01, n_modules),
                           label_counts[i],
                           color="black",
                           s=0.5
                           )
        elif style=="bar_error":
            ax.errorbar(
                x=np.arange(0, n_modules, 1),
                y=bar_heights,
                yerr=bar_sems,
                linestyle="none",linewidth=0.6,
                color="black",capsize=1,markeredgewidth=0.75
            )
    elif style=="points":
        ax.errorbar(
            x=np.arange(0, n_modules, 1),
            y=bar_heights,
            yerr=bar_sems,
            color=cmap([0.1]),
            linestyle="none",
            marker="o",markersize=2.5,linewidth=0.75,
            capsize=2,markeredgewidth=0.75
        )
    ax.set_xlabel(config["data_source"] + ' Pose Label')
    ax.set_ylabel('Frequency')
    ax.set_xticks(np.arange(0, n_modules, 1))
    ax.tick_params(axis='x', rotation=90, labelsize=plt.rcParams['font.size'] * 0.5, pad=2)
    plt.tight_layout()
    return fig

def plot_module_usage_subgroups(config, labels_df, start, stop, figW=6, figH=3,
                                style="bar_scatter",legend_pos="inside", cmap="viridis_r"):
    """
    Plots the frequency of the pose modules occurring by group in the labels dataframe output by the label_counter function.

    :param labels_df: dataframe output from analysis.label_counter
    :param start: time in seconds to start dataframe from.
    :param stop: time in seconds to stop dataframe at.
    :param fps: frames per second of recording.
    :param figW: figure width
    :param figH:
    :param style:
    :return:
    """
    #To get groupnames in order
    fps = int(config["fps"])
    groupnames = []
    added_groupnames = set()
    for item in [header[0] for header in labels_df.columns]:
        if item not in added_groupnames:
            groupnames.append(item)
            added_groupnames.add(item)
    n_groups=len(groupnames)

    # Frames
    start_frame = start * fps
    stop_frame = stop * fps
    labels_df = labels_df[start_frame:stop_frame]
    total_frames = stop_frame - start_frame
    labels_flat = np.array(labels_df)
    labels_flat = [item for sublist in labels_flat for item in sublist]
    modules = np.unique(labels_flat)
    modules_int=[]
    for module in modules:
        if isinstance(module, str):
            modules_int.append(module)
        else:
            modules_int.append(int(module))
    modules=np.array(modules_int)
    n_modules = len(modules)

    # Label counting
    label_counts = []
    for g in range(n_groups):
        group_g_n=np.sum([item[0]==groupnames[g] for item in labels_df.columns])
        label_counts_i = np.zeros([group_g_n, n_modules])
        for i in range(group_g_n):
            for m in range(n_modules):
                module = modules[m]
                try:
                    module = np.int64(module)
                except ValueError:
                    pass
                label_counts_i[i, m] = np.count_nonzero \
                                           (labels_df[groupnames[g]][
                                                [labels_df[groupnames[g]].columns[i]]] == module) / total_frames
        label_counts.append(label_counts_i)

    bar_heights = np.zeros([n_groups, n_modules])
    bar_sems = np.zeros([n_groups, n_modules])

    for g in range(n_groups):
        bar_heights[g, :] = np.mean(label_counts[g], axis=0)
        bar_sems[g, :] = np.std(label_counts[g], axis=0)/np.sqrt(label_counts[g].shape[0])
    fig, ax = plt.subplots(figsize=(figW, figH), dpi=100)
    scale = 1 / (n_groups + .7)
    cmap = plt.get_cmap(cmap)
    colors = [cmap([i]) for i in np.linspace(0,1,n_groups)]
    if ((style=="bar_scatter") or (style=="bar_error")):
        if style=="bar_scatter":
            for g in range(n_groups):
                for i in range(len(label_counts[g])):
                    ax.scatter(
                        np.arange(0 + scale * g, n_modules + scale * g, 1) + np.random.normal(0, 0.1 * scale, n_modules),
                        label_counts[g][i],
                        color="black",
                        s=0.5
                        )
            bar_alpha=0.5
        elif style=="bar_error":
            for g in range(n_groups):
                ax.errorbar(
                    x=np.arange(0 + scale * g, n_modules + scale * g, 1),
                    y=bar_heights[g],
                    yerr=bar_sems[g],
                    linestyle="none",linewidth=0.6,
                    color="black",capsize=0.3,markeredgewidth=0.75,alpha=0.8
                )
            bar_alpha=0.85
        for g in range(n_groups):
            ax.bar(
                x=np.arange(0 + scale * g, n_modules + scale * g, 1),
                height=bar_heights[g],
                width=scale,
                alpha=bar_alpha,
                color=colors[g],
                label=groupnames[g]
            )
    elif style=="points":
        for g in range(n_groups):
            ax.errorbar(
                x=np.arange(0, n_modules, 1),
                y=bar_heights[g],
                yerr=bar_sems[g],
                color=colors[g],
                label=groupnames[g],
                linestyle="none",
                marker="o",markersize=4,linewidth=0.75,
                capsize=2,markeredgewidth=0.75
            )
    if legend_pos == "inside":
        ax.legend()
    elif legend_pos == "outside":
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    ax.set_xlabel(config["data_source"] + ' Pose Label')
    ax.set_ylabel('Frequency')
    all_num_modules = np.sum([isinstance(module,str) for module in modules])==0
    if ((all_num_modules) and (n_modules>=20)):
        xticks=np.arange(0, n_modules, 5, dtype=int)
        ax.set_xticks(xticks)
        ax.set_xticklabels(modules[xticks])
    else:
        ax.set_xticks(np.arange(0, n_modules, 1))
        ax.set_xticklabels(modules)
    ax.tick_params(axis='x', rotation=90, labelsize=plt.rcParams['font.size'] * 0.2 * n_groups, pad=2)
    plt.tight_layout()
    return fig

def plot_module_usage_stacked(config, labels_df, start, stop, figW=6, figH=3,cmap="viridis_r",title=None,alt_xticks=None):
    """
    Plots the frequency of the pose modules occurring by group in the labels dataframe output by the label_counter function.

    :param labels_df: dataframe output from analysis.label_counter
    :param start: time in seconds to start dataframe from.
    :param stop: time in seconds to stop dataframe at.
    :param fps: frames per second of recording.
    :param figW: figure width
    :param figH:
    :param style:
    :return:
    """
    #To get groupnames in order
    fps = int(config["fps"])
    groupnames = []
    added_groupnames = set()
    for item in [header[0] for header in labels_df.columns]:
        if item not in added_groupnames:
            groupnames.append(item)
            added_groupnames.add(item)
    n_groups=len(groupnames)

    # Frames
    start_frame = start * fps
    stop_frame = stop * fps
    labels_df = labels_df[start_frame:stop_frame]
    total_frames = stop_frame - start_frame
    labels_flat = np.array(labels_df)
    labels_flat = [item for sublist in labels_flat for item in sublist]
    modules = np.unique(labels_flat)
    n_modules = len(modules)

    # Label counting
    label_counts = []
    for g in range(n_groups):
        group_g_n=np.sum([item[0]==groupnames[g] for item in labels_df.columns])
        label_counts_i = np.zeros([group_g_n, n_modules])
        for i in range(group_g_n):
            for m in range(n_modules):
                module = modules[m]
                try:
                    module = np.int64(module)
                except ValueError:
                    pass
                label_counts_i[i, m] = np.count_nonzero \
                                           (labels_df[groupnames[g]][
                                                [labels_df[groupnames[g]].columns[i]]] == module) / total_frames
        label_counts.append(label_counts_i)

    bar_heights = np.zeros([n_groups, n_modules])
    bar_sems = np.zeros([n_groups, n_modules])

    for g in range(n_groups):
        bar_heights[g, :] = np.mean(label_counts[g], axis=0)
        bar_sems[g, :] = np.std(label_counts[g], axis=0)/np.sqrt(label_counts[g].shape[0])
    fig, ax = plt.subplots(figsize=(figW, figH), dpi=100)
    cmap = plt.get_cmap(cmap)
    for c in range(len(modules)):
        if c == 0:
            bar_bottom = np.zeros(n_groups)
        ax.bar(np.arange(0, n_groups, 1), bar_heights[:, c], bottom=bar_bottom, align='center', width=0.99)
        ax.spines['top'].set_visible(False)
        bar_bottom += bar_heights[:, c]
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    ax.set_ylabel("Proportion of Time \nSpent in Pose Module")
    ax.set_xlim([-0.5, n_groups - 0.5])
    ax.set_xticks(np.arange(n_groups))
    if alt_xticks is None:
        ax.set_xticklabels(groupnames)
    else:
        ax.set_xticklabels(alt_xticks)
    if title is not None:
        ax.set_title(title)
    plt.legend(modules,bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    return fig


def network_pairwise_comparison(config, labels_df, start, end, groupnames, scaling=1,include_labels=True,cmap="bwr"):
    """
    Plots network depiction of differences in pose module usage and transitions between two subgroups

    :param labels_df: labels_df from label_counter_subgroups
    :param start: time to start at in seconds
    :param end: time to end at in seconds
    :param groupnames: two groups to include in the 1-vs-1 comparison - order is important
    :param fps: frames per second
    :param scaling:
    :param include_labels:
    :param cmap: matplotlib colormap
    :return:
    """
    fps = int(config["fps"])
    fig, ax = plt.subplots(1, 1, figsize=(4,3), dpi=100)
    fig.suptitle(groupnames[0] + " vs. " + groupnames[1], fontsize=16)

    t_min = -3
    t_max = 3

    #Count label occurences
    start_frame = start * fps
    stop_frame = end * fps
    labels_df = labels_df[start_frame:stop_frame]
    total_frames = stop_frame - start_frame
    labels_flat = np.array(labels_df)
    labels_flat = [item for sublist in labels_flat for item in sublist]
    modules = np.unique(labels_flat)
    n_modules = len(modules)
    label_counts = []
    for g in range(2):
        group_g_n=np.sum([item[0]==groupnames[g] for item in labels_df.columns])
        label_counts_i = np.zeros([group_g_n, n_modules])
        for i in range(group_g_n):
            for m in range(n_modules):
                label_counts_i[i, m] = np.count_nonzero \
                                           (labels_df[groupnames[g]][
                                                [labels_df[groupnames[g]].columns[i]]] == m) / total_frames
        label_counts.append(label_counts_i)
    modulefreqs_t = np.zeros(n_modules)
    modulefreqs_p = np.zeros(n_modules)
    for m in range(n_modules):
        clust_i_control = [x[m] for x in label_counts[0]]
        clust_i_exp = [x[m] for x in label_counts[1]]
        modulefreqs_t[m] = scipy.stats.ttest_ind(clust_i_control, clust_i_exp).statistic
        modulefreqs_p[m] = scipy.stats.ttest_ind(clust_i_control, clust_i_exp).pvalue
        if modulefreqs_p[m] < 0.05:
            print("P-value for cluster " + str(m) + " difference is " + str(
                round(modulefreqs_p[m], 4)) + "; t score is " + str(round(modulefreqs_t[m], 4)))

    # transitions
    transition_matrix_grp1, transition_matrix_grp2 = transition_counter(labels_df, groupnames)
    transition_t = np.zeros([n_modules, n_modules])
    transition_t_abs = np.zeros([n_modules, n_modules])
    for i in range(n_modules):
        for j in range(n_modules):
            transition_matrix_grp1_ij = [x[i, j] for x in np.array(transition_matrix_grp1)]
            transition_matrix_grp2_ij = [x[i, j] for x in np.array(transition_matrix_grp2)]
            transition_t[i, j] = scipy.stats.ttest_ind(transition_matrix_grp1_ij, transition_matrix_grp2_ij).statistic
            pval_i = scipy.stats.ttest_ind(transition_matrix_grp1_ij, transition_matrix_grp2_ij).pvalue
            if pval_i < 0.05:
                print("P-value for transition from cluster " + str(i) +
                      " to cluster " + str(j) + " is " + str(round(pval_i, 4)) + "; t score is " + str(
                    round(transition_t[i, j], 4)))
            if np.isnan(transition_t[i, j]) == True:
                transition_t[i, j] = 0
            transition_t_abs[i, j] = np.absolute(transition_t[i, j])
    transition_t = np.array(transition_t)
    G = nx.from_numpy_array(transition_t, parallel_edges=True)
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
        colors.append(col_w)
    G = nx.from_numpy_array(transition_t_abs * 5000, parallel_edges=True)
    edges = G.edges()
    pos = nx.circular_layout(G)
    scaling = scaling/n_modules
    weights = [G[u][v]['weight']*scaling*5 for u, v in edges]
    weights = list(np.array(weights) / 4000)
    labels = {}
    plot = nx.draw_circular(G,
                            node_color=modulefreqs_t,
                            cmap=plt.get_cmap(cmap),
                            vmin=-3,
                            vmax=3,
                            node_size=np.absolute(modulefreqs_t) * 1000 * scaling,
                            edge_color=colors,
                            edgecolors=None,
                            with_labels=True,  # Keep this as True to display node labels
                            labels=labels,
                            font_size=500*scaling,
                            font_weight="bold",
                            font_color="white",
                            width=weights)

    if include_labels==True:
        for m in range(n_modules):
            x, y = pos[m]
            label = str(m)
            label_font_size = 2 + 30 * scaling * (abs(modulefreqs_t[m]) + 1) / 2
            plt.text(x, y, label, color="white", fontsize=label_font_size, fontweight="bold", ha="center", va="center")

    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(cmap), norm=plt.Normalize(vmin=t_min, vmax=t_max))
    sm.set_array([])
    cbar = plt.colorbar(sm, shrink=0.7)
    cbar.set_label('t-score')
    plt.margins(x=0.4, y=0.4)
    return fig


def transition_counter(labels_df, groupnames):
    """
    Generates a Pandas dataframe containing transitions within the labels df.

    :param labels_df:
    :param groupnames:
    :return:
    """
    unique_values = np.unique(labels_df.values)
    transition_matrix_grp1 = []
    transition_matrix_grp2 = []
    for g in range(len(groupnames)):
        labels_df_g = labels_df[groupnames[g]]
        for c in range(labels_df_g.shape[1]):
            transition_matrix_g = pd.DataFrame(0, index=unique_values, columns=unique_values)
            for i in range(len(labels_df_g[list(labels_df_g)[c]]) - 1):
                from_value = labels_df_g[list(labels_df_g)[c]][i]
                to_value = labels_df_g[list(labels_df_g)[c]][i + 1]
                transition_matrix_g.at[from_value, to_value] += 1
            np.fill_diagonal(transition_matrix_g.values, 0)
            if g == 0:
                transition_matrix_grp1.append(transition_matrix_g)
            if g == 1:
                transition_matrix_grp2.append(transition_matrix_g)
    return transition_matrix_grp1, transition_matrix_grp2

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def SandPlotClusterFrequency_OverTime(config,
                                      labels_df,
                                      start,
                                      time_per_block,
                                      n_blocks,
                                      figW=7, figH=3,
                                      posenames=None,
                                      title=None,
                                      convolve=4,
                                      legend=True,
                                      plottype='area'):
    """
    Plots usage of pose modules over time within a session

    :param labels_df:
    :param start: time to start at in seconds
    :param time_per_block: time per block in seconds
    :param fps: frames per second
    :param n_blocks:
    :param figW:
    :param figH:
    :param posenames:
    :param title:
    :param convolve:
    :param legend:
    :param plottype:
    :param saveplots:
    :param savename:
    :return:
    """
    fps = int(config["fps"])
    n_samples = len(labels_df.columns)
    labels_flat = np.array(labels_df)
    labels_flat = [item for sublist in labels_flat for item in sublist]
    modules = np.unique(labels_flat)
    n_modules = len(modules)
    block_labels = np.zeros([n_samples, n_modules, n_blocks])
    block_labels_normal = np.zeros([n_samples, n_modules, n_blocks])
    frames_per_block = int(fps * time_per_block)
    for i in range(n_samples):
        for b in range(n_blocks):
            new_start = start + b * frames_per_block
            x = labels_df[labels_df.columns[i]][new_start:new_start + frames_per_block]
            for m, module in enumerate(modules):
                block_labels[i, m, b] = np.count_nonzero(x == module)
                block_labels_normal[i, m, b] = np.count_nonzero(x == module) / frames_per_block

    bar_heights = np.zeros([n_blocks, n_modules])
    bar_heights_normal = np.zeros([n_blocks, n_modules])
    bar_stds = np.zeros([n_blocks, n_modules])
    bar_stds_normal = np.zeros([n_blocks, n_modules])

    for r in range(n_blocks):
        for q in range(n_modules):
            bar_heights[r, q] = np.mean(block_labels[:, q, r])
            bar_stds[r, q] = np.std(block_labels[:, q, r])
            bar_heights_normal[r, q] = np.mean(block_labels_normal[:, q, r])
            bar_stds_normal[r, q] = np.std(block_labels_normal[:, q, r])

    if plottype == 'line':
        fig, ax = plt.subplots(figsize=(figW, figH), dpi=100)
        scale = 1 / (n_blocks + .7)
        for r in range(n_modules):
            plt.plot(moving_average(bar_heights_normal[:, r], convolve))
            plt.fill_between(np.arange(0, n_blocks - convolve + 1, 1),
                             moving_average(bar_heights_normal[:, r] + bar_stds_normal[:, r] / np.sqrt(n_samples),
                                            convolve),
                             moving_average(bar_heights_normal[:, r] - bar_stds_normal[:, r] / np.sqrt(n_samples),
                                            convolve),
                             alpha=0.2)
        if legend == True:
            if posenames == None:
                if np.sum([type(m)!=int for m in modules])>0:
                    posenames=modules
                else:
                    posenames = []
                    for i in range(n_modules):
                        posenames.append("Pose " + str(i))
                ax.legend(posenames, loc="upper right", bbox_to_anchor=(1.25, 1.0))
            else:
                ax.legend(posenames, loc="upper right", bbox_to_anchor=(1.25, 1.0))

        scale = 1 / (n_modules + .7)
        ax.set_xlabel('Time (minutes)')
        ax.set_ylabel('Moving Average Proportion of \nTime Spent in Pose')
        ax.set_ylim([-0.03, 1.03])

    elif plottype == 'area':
        fig, ax = plt.subplots(figsize=(figW, figH), dpi=100)
        bar_heights_moving_average = np.zeros([n_blocks - convolve + 1, n_modules])
        for r in range(n_modules):
            bar_heights_moving_average[:, r] = moving_average(bar_heights_normal[:, r], convolve)
        plt.stackplot(np.arange(0, bar_heights_moving_average.shape[0], 1), bar_heights_moving_average.T)
        plt.xlim(0, n_blocks - convolve)
        plt.ylim(0, 1)
        plt.xlabel('Time (minutes)')
        plt.ylabel('Moving Average Proportion \nof Time Spent in Pose')

        if legend == True:
            if posenames == None:
                if np.sum([type(m)!=int for m in modules])>0:
                    posenames=modules
                else:
                    posenames = []
                    for i in range(n_modules):
                        posenames.append("Pose " + str(i))
                ax.legend(posenames, loc="upper right", bbox_to_anchor=(1.25, 1.0))
            else:
                ax.legend(posenames, loc="upper right", bbox_to_anchor=(1.25, 1.0))

    plt.tight_layout()

    if title!=None:
        ax.set_title(title)

    return fig


def plot_dist_bins(dist_df, cmap="viridis", plottype="band", figW=6, figH=3):
    """
    Plots distance of a keypoint either over time or in bins from dist_df (output of analysis.dist_df_subgroups)

    :param dist_df: dist_df output from analysis.dist_df_subgroups
    :param cmap: matplotlib colormap
    :param plottype: type of plot ("band", "errorbar", or "bar" if no timebins)
    :param figW: figure width
    :param figH: figure height
    :return:
    """
    fig, ax = plt.subplots(figsize=(figW, figH), dpi=100)
    groups = pd.Series([i[0] for i in dist_df.columns]).unique()
    n_groups = len(groups)
    if dist_df.shape[0] == 1:
        plottype = "bar"
    cmap = plt.get_cmap(cmap)
    colors = [cmap([i]) for i in np.linspace(0, 1, n_groups)]
    xticks = dist_df.index
    for g, group in enumerate(groups):
        sub_df = dist_df[group]
        group_mean = sub_df.mean(axis=1)
        group_sem = sub_df.std(axis=1) / np.sqrt(sub_df.shape[1])
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

def make_and_plot_ellipse(mean, cov, color, label=None):
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]) * 180 / np.pi
    ell = Ellipse(mean, width=2 * np.sqrt(eigenvalues[0]), height=2 * np.sqrt(eigenvalues[1]),
                   angle=angle, facecolor=color, alpha=0.25, label=label,edgecolor="none")
    plt.gca().add_patch(ell)

def plot_lda(config, lda_result, figW=4, figH=3, titletype="informative", cmap="jet", marker_dict=None,
             draw_ellipse=False,alt_labels=None):
    """
    Plot LDA embeddings

    :param config: config file
    :param lda: lda from lda_labels_timebins()
    :param lda_embeddings: embeddings from lda_labels_timebins()
    :param group_labels: group_labels from lda_labels_timebins()
    :param nbins: number of bins from lda_labels_timebins()
    :param binsize: binsize
    :param figW: width of the figure
    :param figH: height of the figure
    :param titletype: type of title - options are "informative", "uninformative"
    :param cmap: matplotlib colormap OR a dictionary with keys corresponding to group names and values corresponding to colors
    :param marker_dict: dictionary with keys corresponding to group names and values corresponding to colors
    :param draw_ellipse: whether to simulate data and draw an ellipse fitted to the class of the data or not
    :param alt_labels: dictionary giving alternate labels for each subgroup key
    :return: figure
    """
    selected_subgroups=list(lda_result.group_dict.keys())
    n_groups=len(selected_subgroups)
    flip_group_dict = {v: k for k, v in lda_result.group_dict.items()}
    if type(cmap) != dict:
        cmap = plt.get_cmap(cmap)
        colors = [cmap([i]) for i in np.linspace(0,1,len(selected_subgroups))]
    else:
        colors = []
        for r in range(n_groups):
            group = flip_group_dict[r]
            colors.append(cmap[group])
    if marker_dict is not None:
        markers = []
        for r in range(n_groups):
            group = flip_group_dict[r]
            markers.append(marker_dict[group])
    else:
        markers = ["o"]*n_groups
    LD1 = lda_result.lda_embeddings[:, 0]
    LD2 = lda_result.lda_embeddings[:, 1]
    fig = plt.figure(figsize=(figW, figH), dpi=100)
    for r in range(n_groups):
        label_in_legend = False
        for i in range(len(lda_result.group_labels)):
            if lda_result.group_labels[i] == r:
                LD1_i = LD1[i]
                LD2_i = LD2[i]
                if alt_labels is None:
                    label = flip_group_dict[r]
                else:
                    label = alt_labels[flip_group_dict[r]]
                if label_in_legend==True:
                    plt.scatter(LD1_i, LD2_i, c=colors[r], s=23, marker=markers[r])
                else:
                    plt.scatter(LD1_i, LD2_i, c=colors[r], s=23, marker=markers[r],label=label)
                    label_in_legend=True
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlabel("LD1 (" + str(int(1000 * lda_result.lda.explained_variance_ratio_[0]) / 10) + "% Variance Explained)")
    plt.ylabel("LD2 (" + str(int(1000 * lda_result.lda.explained_variance_ratio_[1]) / 10) + "% Variance Explained)")
    if titletype == "uninformative":
        plt.title("Linear Discriminant Analysis\n with Time Bins", fontweight="bold")
    elif titletype == "informative":
        plt.title(str("Linear Discriminant Analysis\n with " + str(lda_result.nbins) + " " + str(lda_result.binsize / 60) + "-min time bins"),
                  fontweight="bold")
    if draw_ellipse==True:
        for r in range(n_groups):
            class_dat = lda_result.label_counts[[i == r for i in lda_result.group_labels], :]
            emb = lda_result.lda.transform(class_dat[:,lda_result.feat_picks])
            mean = np.mean(emb, axis=0)
            cov = np.cov(emb, rowvar=False)
            make_and_plot_ellipse(mean, cov, color=colors[r], label=flip_group_dict[r])
    plt.tight_layout()
    return fig


def plot_lda_weights(config, lda_result, n_modules, hide_nofeat_mods=False, remap=False, figW=6, figH=4):
    """

    :param lda_result:
    :param n_modules:
    :param hide_nofeat_mods:
    :return:
    """
    weights = lda_result.get_discriminant_weights()
    mods = [f"module{i}" for i in range(n_modules)]
    bins = []
    for b in range(lda_result.nbins):
        binstart = int(b * (lda_result.binsize))
        binstop = int((b + 1) * (lda_result.binsize))
        bins.append(f"t{binstart}-{binstop}")

    figs = []
    for LD in list(weights.columns):
        weights_reshaped = pd.DataFrame(index=bins, columns=mods)
        for bin, row in weights_reshaped.iterrows():
            for mod in weights_reshaped.columns:
                if len(bins) == 1:
                    i = mod
                else:
                    i = mod + "_" + bin
                if i not in weights.index:
                    weights_reshaped.at[bin, mod] = np.nan
                else:
                    weights_reshaped.at[bin, mod] = weights.at[i, LD]
        if hide_nofeat_mods == True:
            weights_reshaped = weights_reshaped.dropna(axis=1, how='all')

        if remap == False:
            fig = plt.figure(figsize=(figW, figH))
            vlim = np.max([np.abs(weights_reshaped.max().max()), np.abs(weights_reshaped.min().min())])
            data = np.array(weights_reshaped, dtype=np.float32)
            plt.imshow(data, cmap="seismic", vmin=-vlim, vmax=vlim, aspect="auto",interpolation=None)
            plt.imshow(np.zeros_like(data) + 0.8, cmap="gray", vmax=1, vmin=0, aspect="auto")
            plt.imshow(data, cmap="seismic", vmin=-vlim, vmax=vlim, aspect="auto",interpolation=None)
            names = [i.replace("module", "") for i in weights_reshaped.columns]
            if len(names) > 50:
                tick_positions = range(0, len(names), 10)
                tick_labels = [names[i] for i in tick_positions]
                plt.xticks(ticks=tick_positions, labels=tick_labels)
            elif len(names) > 20:
                tick_positions = range(0, len(names), 5)
                tick_labels = [names[i] for i in tick_positions]
                plt.xticks(ticks=tick_positions, labels=tick_labels)
            else:
                plt.xticks(ticks=range(weights_reshaped.shape[1]), labels=names)

            binnames = [bin.replace("t", "") for bin in bins]
            if len(binnames) > 10:
                tick_positions = range(0, len(binnames), 5)
                tick_labels=[]
                for t in tick_positions:
                    tick_labels.append(binnames[t])
                plt.yticks(ticks=tick_positions, labels=tick_labels)
            else:
                plt.yticks(ticks=range(len(bins)), labels=[bin.replace("t", "") for bin in bins])
            cbar = plt.colorbar()
            cbar.set_label('Weight')
            plt.ylabel('Time bin (s)')
            plt.xlabel("Modules")
            plt.title(LD)

        elif remap == True:
            vlim = np.max([np.abs(weights_reshaped.max().max()), np.abs(weights_reshaped.min().min())])
            fig = plt.figure(figsize=(figW, figH))

            BORIS_to_pose_mat, BORIS_to_pose_mat_normalized, loss = analysis.BORIS_to_pose(config)
            analysis.make_remappings_from_BORIS(config, None, BORIS_to_pose_mat)

            behaviors = BORIS_to_pose_mat_normalized.index
            mapped_behaviors = {}
            for beh in behaviors:
                beh_mod = []
                beh_mod.extend([i[0][0] for i in config["remappings"] if i[1] == beh])
                if len(beh_mod) > 0:
                    mapped_behaviors[beh] = beh_mod

            widths = []
            for b, beh_i in enumerate(list(mapped_behaviors.keys())):
                mapped_behaviors[beh_i] = ["module" + str(int(i)) for i in mapped_behaviors[beh_i]]
                if hide_nofeat_mods == True:
                    mapped_behaviors[beh_i] = [i for i in mapped_behaviors[beh_i] if i in weights_reshaped.columns]
                if len(mapped_behaviors[beh_i]) > 0:
                    widths.append(len(mapped_behaviors[beh_i]))
                else:
                    del mapped_behaviors[beh_i]
            total_mods = np.sum(widths)
            widths.extend([total_mods / 15])
            widths = widths / np.sum(widths)
            gs = gridspec.GridSpec(1, len(widths), width_ratios=widths)
            for b, beh_i in enumerate(list(mapped_behaviors.keys())):
                ax = plt.subplot(gs[b])
                beh_i_mods = mapped_behaviors[beh_i]
                data = np.array(weights_reshaped[beh_i_mods], dtype=np.float32)
                ax.imshow(data, cmap="seismic", aspect="auto", vmin=-vlim, vmax=vlim,interpolation=None)
                ax.imshow(np.zeros_like(data) + 0.8, cmap="gray", vmax=1, vmin=0, aspect="auto")
                cb = ax.imshow(data, cmap="seismic", aspect="auto", vmin=-vlim, vmax=vlim,interpolation=None)
                ax.set_xticks(ticks=range(len(mapped_behaviors[beh_i])), labels=mapped_behaviors[beh_i])
                ax.set_xlabel(beh_i)
                names = [i.replace("module", "") for i in weights_reshaped[beh_i_mods].columns]
                if total_mods < 30:
                    ax.set_xticks(ticks=range(weights_reshaped[beh_i_mods].shape[1]), labels=names)
                else:
                    ax.set_xticks(ticks=range(weights_reshaped[beh_i_mods].shape[1]), labels=[])

                binnames = [bin.replace("t", "") for bin in bins]
                if b == 0:
                    ax.set_ylabel('Time bin (s)')
                    if len(binnames) > 10:
                        tick_positions = range(0, len(binnames), 5)
                        tick_labels=[]
                        for t in tick_positions:
                            tick_labels.append(binnames[t])
                        ax.set_yticks(ticks=tick_positions, labels=tick_labels)
                    else:
                        ax.set_yticks(ticks=range(len(bins)), labels=[bin.replace("t", "") for bin in bins])
                else:
                    ax.set_yticks(ticks=range(len(bins)), labels=[])
            plt.suptitle(LD)
            cbar_ax = plt.subplot(gs[-1])
            cbar = plt.colorbar(cb, cax=cbar_ax)
            cbar.set_label("Weight")
        plt.tight_layout()
        figs.append(fig)
    return figs

def plot_conf_mat(lda_result, figW=2.5,figH=2.5,cmap="Greens",
                  alt_title=False,rotate_xticks=False,alt_labels=None):
    """
    Generate a confusion matrix plot

    :param confusion: the confusion matrix from sklearn
    :param class_num: the classes (as integers)
    :param class_labels: the classes (string names)
    :param figW: figure width
    :param figH: figure height
    :param cmap: colormap
    :param alt_title: title other than confusion matrix
    :param rotate_xticks: whether or not to rotate xticks
    :return:
    """
    fig = plt.figure(figsize=(figW, figH), dpi=100)
    plt.imshow(lda_result.loocv_confmat, cmap=cmap)
    class_labels=list(lda_result.group_dict.keys())
    if alt_labels is not None:
        class_labels_ticks=class_labels.copy()
        for i in range(len(class_labels_ticks)):
            class_labels_ticks[i]=alt_labels[class_labels[i]]
    else:
        class_labels_ticks=class_labels.copy()

    ticks=[lda_result.group_dict[key] for key in class_labels]
    if rotate_xticks==True:
        plt.xticks(ticks, class_labels_ticks,rotation=90)
    else:
        plt.xticks(ticks, class_labels_ticks)
    plt.yticks(ticks, class_labels_ticks)
    for i in range(len(class_labels)):
        for j in range(len(class_labels)):
            plt.text(j, i, str(int(lda_result.loocv_confmat[i, j])), ha='center', va='center', color='black')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    if alt_title==False:
        plt.title('Confusion Matrix')
    else:
        plt.title(alt_title)
    plt.tight_layout()
    return fig


# def plot_conf_mat(confusion, class_num, class_labels, figW=2.5,figH=2.5,cmap="Greens",alt_title=False):
#     """
#     Generate a confusion matrix plot
#     :param confusion: the confusion matrix from sklearn
#     :param class_num: the classes (as integers)
#     :param class_labels: the classes (string names)
#     :param figW: figure width
#     :param figH: figure height
#     :param cmap: colormap
#     :param alt_title: title other than confusion matrix
#     :return:
#     """
#     fig = plt.figure(figsize=(figW, figH), dpi=100)
#     plt.imshow(confusion, cmap=cmap)
#     plt.xticks(class_num, class_labels)
#     plt.yticks(class_num, class_labels)
#     for i in range(len(class_labels)):
#         for j in range(len(class_labels)):
#             plt.text(j, i, str(confusion[i, j]), ha='center', va='center', color='black')
#     plt.xlabel('Predicted Dose Label')
#     plt.ylabel('True Dose Label')
#     if alt_title==False:
#         plt.title('Confusion Matrix')
#     else:
#         plt.title(alt_title)
#     plt.tight_layout()
#     return fig

def plot_pc_weights(pca,cmap="PuOr"):
    """
    Plot PCA weights

    :param pca: pca object from sklearn
    :return: fig
    """
    components = pca.components_
    fig = plt.figure(figsize=(4,2),dpi=100)
    pc_labels=["PC"+str(i+1) for i in range(pca.components_.shape[0])]
    plt.imshow(components,cmap=cmap,vmin=-1,vmax=1)
    plt.title("Principle Component Weights")
    plt.xticks(np.arange(0,pca.components_.shape[1],1))
    plt.xlabel("Pose Modules")
    plt.yticks([0,1],labels=pc_labels)
    plt.colorbar(cmap=cmap)
    plt.tight_layout()
    return fig

def BORIS_to_pose_matrix_plot(config, boris_to_pose_output, figW=4, figH=2.5, cmap="Greens",outline_top_match=True):
    fig, ax = plt.subplots(figsize=(figW, figH), dpi=100)
    plt.imshow(boris_to_pose_output.to_numpy(dtype='float'),cmap=cmap,aspect="auto")
    data = boris_to_pose_output.to_numpy(dtype='float')
    num_cols = data.shape[1]
    if outline_top_match==True:
        for col in range(num_cols):
            max_row = np.argmax(data[:, col])
            if np.sum(data[:, col]==data[max_row, col])==1:
                rect = Rectangle((col - 0.5, max_row - 0.5), 1, 1, edgecolor='purple', facecolor='none', linewidth=0.8)
                ax.add_patch(rect)
    if len(boris_to_pose_output.columns>=20):
        xticks=np.arange(0,np.max(boris_to_pose_output.columns),5,dtype=int)
        xticklabels=np.array(list(boris_to_pose_output.columns),dtype=int)[xticks]
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

