import scipy.io
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import pandas as pd
import networkx as nx
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']


def plot_module_usage(config,labels_df,start,stop,figW=4,figH=2,style="bar_scatter",cmap="jet"):
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
    fps = config["fps"]
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
            label_counts[i,m] = np.count_nonzero \
                (labels_df[[labels_df.columns[i]]] == m) /total_frames
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
                linestyle="none",linewidth=0.75,
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
    ax.set_xlabel(config["data_type"] + ' Pose Label')
    ax.set_ylabel('Frequency')
    ax.set_xticks(np.arange(0, n_modules, 1))
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
    fps = config["fps"]
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
                label_counts_i[i, m] = np.count_nonzero \
                                           (labels_df[groupnames[g]][
                                                [labels_df[groupnames[g]].columns[i]]] == m) / total_frames
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
                    linestyle="none",linewidth=0.75,
                    color="black",capsize=1,markeredgewidth=0.75
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

    ax.set_xlabel(config["data_type"] + ' Pose Label')
    ax.set_ylabel('Frequency')
    ax.set_xticks(np.arange(0, n_modules, 1))
    plt.tight_layout()
    return fig


def network_pairwise_comparison(config, labels_df, start, end, groupnames, scaling=0.2,include_labels=True,cmap="bwr"):
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
    fps = config["fps"]
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
    weights = [G[u][v]['weight'] for u, v in edges]
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
                            font_size=100 * scaling,
                            font_weight="bold",
                            font_color="white",
                            width=weights)

    if include_labels==True:
        for m in range(n_modules):
            x, y = pos[m]
            label = str(m)
            label_font_size = 24 * scaling * (abs(modulefreqs_t[m]) + 1) / 2
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
                                      figW=10, figH=3,
                                      posenames=None,
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
    fps = config["fps"]
    n_samples = len(labels_df.columns)
    labels_flat = np.array(labels_df)
    labels_flat = [item for sublist in labels_flat for item in sublist]
    clusts = np.unique(labels_flat)
    n_clust = len(clusts)
    block_labels = np.zeros([n_samples, n_clust, n_blocks])
    block_labels_normal = np.zeros([n_samples, n_clust, n_blocks])
    frames_per_block = int(fps * time_per_block)
    for i in range(n_samples):
        for b in range(n_blocks):
            new_start = start + b * frames_per_block
            x = labels_df[labels_df.columns[i]][new_start:new_start + frames_per_block]
            for q in range(n_clust):
                block_labels[i, q, b] = np.count_nonzero(x == q)
                block_labels_normal[i, q, b] = np.count_nonzero(x == q) / frames_per_block

    bar_heights = np.zeros([n_blocks, n_clust])
    bar_heights_normal = np.zeros([n_blocks, n_clust])
    bar_stds = np.zeros([n_blocks, n_clust])
    bar_stds_normal = np.zeros([n_blocks, n_clust])

    for r in range(n_blocks):
        for q in range(n_clust):
            bar_heights[r, q] = np.mean(block_labels[:, q, r])
            bar_stds[r, q] = np.std(block_labels[:, q, r])
            bar_heights_normal[r, q] = np.mean(block_labels_normal[:, q, r])
            bar_stds_normal[r, q] = np.std(block_labels_normal[:, q, r])

    if plottype == 'line':
        fig, ax = plt.subplots(figsize=(figW, figH), dpi=100)
        scale = 1 / (n_blocks + .7)
        for r in range(n_clust):
            plt.plot(moving_average(bar_heights_normal[:, r], convolve))
            plt.fill_between(np.arange(0, n_blocks - convolve + 1, 1),
                             moving_average(bar_heights_normal[:, r] + bar_stds_normal[:, r] / np.sqrt(n_samples),
                                            convolve),
                             moving_average(bar_heights_normal[:, r] - bar_stds_normal[:, r] / np.sqrt(n_samples),
                                            convolve),
                             alpha=0.2)
        if legend == True:
            if posenames == None:
                posenames = []
                for i in range(n_clust):
                    posenames.append("Pose " + str(i))
                ax.legend(posenames, bbox_to_anchor=(1.1, 1.05))
            else:
                ax.legend(posenames, bbox_to_anchor=(1.1, 1.05))

        scale = 1 / (n_clust + .7)
        ax.set_xlabel('Time (minutes)')
        ax.set_ylabel('Moving Average Proportion of \nTime Spent in Pose')
        ax.set_ylim([-0.03, 1.03])

    elif plottype == 'area':
        fig, ax = plt.subplots(figsize=(figW, figH), dpi=100)
        bar_heights_moving_average = np.zeros([n_blocks - convolve + 1, n_clust])
        for r in range(n_clust):
            bar_heights_moving_average[:, r] = moving_average(bar_heights_normal[:, r], convolve)
        plt.stackplot(np.arange(0, bar_heights_moving_average.shape[0], 1), bar_heights_moving_average.T)
        plt.xlim(0, n_blocks - convolve)
        plt.ylim(0, 1)
        plt.xlabel('Time (minutes)')
        plt.ylabel('Moving Average Proportion \nof Time Spent in Pose')

        if legend == True:
            if posenames == None:
                posenames = []
                for i in range(n_clust):
                    posenames.append("Pose " + str(i))
                ax.legend(posenames, bbox_to_anchor=(1.1, 1.05))
            else:
                ax.legend(posenames, bbox_to_anchor=(1.1, 1.05), loc="upper left")
        plt.tight_layout()
        return fig



def plot_lda(config, lda, lda_embeddings, group_labels, nbins, binsize, selected_subgroups="all",
             figW=4, figH=3, titletype="informative", cmap="jet"):
    """
    Plot LDA embeddings
    :param config: config file
    :param lda: lda from lda_labels_timebins()
    :param lda_embeddings: embeddings from lda_labels_timebins()
    :param group_labels: group_labels from lda_labels_timebins()
    :param nbins: number of bins from lda_labels_timebins()
    :param binsize: binsize
    :param selected_subgroups: which subgroups to plot
    :param figW: width of the figure
    :param figH: height of the figure
    :param titletype: type of title - options are "informative", "uninformative"
    :param cmap: matplotlib colormap
    :return: figure
    """
    if selected_subgroups=="all":
        selected_subgroups=list(config["subgroups"].keys())
    n_groups=len(selected_subgroups)
    cmap = plt.get_cmap(cmap)
    colors = [cmap([i]) for i in np.linspace(0,1,len(selected_subgroups))]
    LD1 = lda_embeddings[:, 0]
    LD2 = lda_embeddings[:, 1]
    fig = plt.figure(figsize=(figW, figH), dpi=100)
    for i in range(len(group_labels)):
        for r in range(n_groups):
            if group_labels[i] == r:
                LD1_i = LD1[i]
                LD2_i = LD2[i]
                plt.scatter(LD1_i, LD2_i, c=colors[r], s=23, marker="o")
    leg = plt.legend(selected_subgroups, bbox_to_anchor=(1.05, 1), loc='upper left')
    for i in range(n_groups):
        leg.legendHandles[i].set_color(colors[i])
    plt.xlabel("LD1 (" + str(int(1000 * lda.explained_variance_ratio_[0]) / 10) + "% Variance Explained)")
    plt.ylabel("LD2 (" + str(int(1000 * lda.explained_variance_ratio_[1]) / 10) + "% Variance Explained)")
    if titletype == "uninformative":
        plt.title("Linear Discriminant Analysis\n with Time Bins", fontweight="bold")
    elif titletype == "informative":
        plt.title(str("Linear Discriminant Analysis\n with " + str(nbins) + " " + str(binsize / 60) + "-min time bins"),
                  fontweight="bold")
    plt.tight_layout()
    return fig

def confusion_matrix_plot(config, labels_df):
    print("Coming soon!")
    # TODO:: add all-purpose confusion matrix plot function

def pose_to_BORIS_plot(config, labels_df):
    print("Coming soon!")
    # TODO:: add function for plotting pose to BORIS comparison matrix

