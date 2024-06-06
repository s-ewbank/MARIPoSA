import numpy as np
import pandas as pd
import os
from sklearn.tree import plot_tree
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression as LR

def label_counter_nosubgroups(config, start, stop):
    """
    Generates a Pandas dataframe containing the labels for every frame in the specified time range for the video paths in defined groups.

    :param config: the config
    :param start: time in seconds to start dataframe from.
    :param stop: time in seconds to stop dataframe at.
    :param fps: frames per second of recording.
    :return:
    """
    fps = config["fps"]
    start_frame = start * fps
    stop_frame = stop * fps
    count = 0
    header = []
    if config["data_type"]=="B-SOiD":
        label_paths=[i for i in config["project_files"] if "labels_" in i]
        for i in range(len(label_paths)):
            header.append(label_paths[i])
            labels_i = np.loadtxt(
                str(config["data_directory"] + "/" + label_paths[i]),
                delimiter=",", skiprows=3, usecols=1)[start_frame:stop_frame]
            if i == 0:
                labels_df = pd.DataFrame(labels_i, columns=[header])
            else:
                labels_df.insert(loc=count, value=labels_i, column=label_paths[i])
            count = count + 1
    elif config["data_type"]=="Keypoint-MoSeq":
        label_paths=config["project_files"]
        for i in range(len(label_paths)):
            header.append(label_paths[i])
            labels_i = np.loadtxt(
                str(config["data_directory"] + "/" + label_paths[i]),
                delimiter=",", skiprows=1, usecols=0)[start_frame:stop_frame]
            if i == 0:
                labels_df = pd.DataFrame(labels_i, columns=[header])
            else:
                labels_df.insert(loc=count, value=labels_i, column=label_paths[i])
            count = count + 1
    elif config["data_type"]=="VAME":
        data_directory = config["data_directory"]
        model_path = ""
        model_path = model_path + "/" + os.listdir(data_directory + "/" + os.listdir(data_directory)[0])[0]
        model_path = model_path + "/" + os.listdir(data_directory + "/" + os.listdir(data_directory)[0] + model_path)[
            0] + "/"
        for f in range(len(config["project_files"])):
            header.append(config["project_files"][f])
            datstr = [i for i in os.listdir(data_directory + "/" + config["project_files"][f] + model_path) if "_label_" in i][0]
            labels_i = np.load(data_directory + "/" + config["project_files"][f] + model_path + datstr)[
                       start_frame:stop_frame]
            if f == 0:
                labels_df = pd.DataFrame(labels_i, columns=[header])
            else:
                labels_df.insert(loc=count, value=labels_i, column=config["project_files"][f])
            count = count + 1

    #Get number of modules
    labels_flat = np.array(labels_df)
    labels_flat = [item for sublist in labels_flat for item in sublist]
    modules = np.unique(labels_flat)
    n_modules = len(modules)
    return labels_df, n_modules

def label_counter_subgroups(config, start, stop, selected_subgroups="all"):
    """
    Generates a Pandas dataframe containing the labels for every frame in the specified time range for the video paths in defined groups.

    :param config: the config
    :param start: time in seconds to start dataframe from.
    :param stop: time in seconds to stop dataframe at.
    :param fps: frames per second of recording.
    :param selected_subgroups: frames per second of recording.
    :return:
    """
    fps = config["fps"]
    start_frame = start * fps
    stop_frame = stop * fps
    if selected_subgroups=="all":
        selected_subgroups=list(config["subgroups"].keys())
    n_groups = len(selected_subgroups)
    count = 0
    header1 = []
    header2 = []
    if config["data_type"]=="B-SOiD":
        for g in range(n_groups):
            label_paths_g = [i for i in config["subgroups"][selected_subgroups[g]] if "labels_" in i]
            for i in range(len(label_paths_g)):
                header = label_paths_g[i]
                header2.append(header)
                header1.append(selected_subgroups[g])
                labels_i = np.loadtxt(
                    str(config["data_directory"] + "/" + label_paths_g[i]),
                    delimiter=",", skiprows=3, usecols=1)[start_frame:stop_frame]
                if i == 0 and g == 0:
                    labels_df = pd.DataFrame(labels_i, columns=[header])
                else:
                    labels_df.insert(loc=count, value=labels_i, column=header)
                count = count + 1
        labels_df.columns = [header1, header2]
    elif config["data_type"]=="Keypoint-MoSeq":
        for g in range(n_groups):
            label_paths_g = config["subgroups"][selected_subgroups[g]]
            for i in range(len(label_paths_g)):
                header = label_paths_g[i]
                header2.append(header)
                header1.append(selected_subgroups[g])
                labels_i = np.loadtxt(
                    str(config["data_directory"] + "/" + label_paths_g[i]),
                    delimiter=",", skiprows=1, usecols=0)[start_frame:stop_frame]
                if i == 0 and g == 0:
                    labels_df = pd.DataFrame(labels_i, columns=[header])
                else:
                    labels_df.insert(loc=count, value=labels_i, column=header)
                count = count + 1
        labels_df.columns = [header1, header2]
    elif config["data_type"]=="VAME":
        data_directory = config["data_directory"]
        model_path = ""
        model_path = model_path + "/" + os.listdir(data_directory + "/" + os.listdir(data_directory)[0])[0]
        model_path = model_path + "/" + os.listdir(data_directory + "/" + os.listdir(data_directory)[0] + model_path)[
            0] + "/"
        for g in range(n_groups):
            label_paths_g = config["subgroups"][selected_subgroups[g]]
            for f in range(len(label_paths_g)):
                header = label_paths_g[f]
                header2.append(header)
                header1.append(selected_subgroups[g])
                datstr = [i for i in os.listdir(data_directory + "/" + label_paths_g[f] + model_path) if "_label_" in i][0]
                labels_i = np.load(data_directory + "/" + label_paths_g[f] + model_path + datstr)[
                           start_frame:stop_frame]
                if f == 0 and g == 0:
                    labels_df = pd.DataFrame(labels_i, columns=[header])
                else:
                    labels_df.insert(loc=count, value=labels_i, column=header)
                count = count + 1
        labels_df.columns = [header1, header2]

    #Get number of modules
    labels_flat = np.array(labels_df)
    labels_flat = [item for sublist in labels_flat for item in sublist]
    modules = np.unique(labels_flat)
    n_modules = len(modules)
    return labels_df, n_modules

def BORIS_to_pose(config):
    #plot pose to whatever wahtever
    boris_directory = config["boris_directory"]
    boris_to_pose_pairings = config["boris_to_pose_pairings"]
    results=None
    for pairing in boris_to_pose_pairings:
        if pairing[0]==None:
            continue
        else:
            config_modulo=config
            config_modulo["project_files"]=[pairing[1]]
            labels_df, n_modules = label_counter_nosubgroups(config_modulo,0,1200)
            boris_i = pd.read_csv(boris_directory + "/" + pairing[0])
            boris_i["frame"] = np.round(boris_i["time"] / (1 / config["fps"]))
            boris_i = boris_i.drop_duplicates(subset='frame', keep='first')
            fr = np.min([len(labels_df), np.max(boris_i["frame"])])
            boris_i=boris_i[0:int(fr)]
            labels_df = labels_df[0:int(fr)]
            boris_i = boris_i.drop(["time", "frame"], axis=1)
            boris_i = boris_i.reset_index(drop=True)
            behaviors = list(boris_i.columns)
            if results is None:
                results = pd.DataFrame(data=np.zeros([n_modules,len(behaviors)]),columns=behaviors,index=np.arange(0,n_modules))
            results_i = pd.DataFrame(columns=behaviors, index=np.arange(0, n_modules))
            for behavior in behaviors:
                mask = boris_i[behavior]>0
                masked_labels = labels_df[mask==True]
                for module in range(n_modules):
                    results_i.at[module,behavior] = np.sum(masked_labels==module).to_numpy()
            results=results+results_i
    results=results.T
    column_sums = results.sum()
    normalized_results = results / column_sums
    return normalized_results

def combine_pose_modules(config, labels_df):
    print("Coming soon!")
    # TODO:: add function for combining pose modules

def lda_labels_timebins(config, labels_df, binsize, selected_subgroups="all", ncomponents=2):
    """
    Function to compute LDA for data in timebins
    :param config:
    :param labels_df:
    :param binsize:
    :param selected_subgroups:
    :param ncomponents:
    :return:
    """
    # if groupnames == None:
    #     groupnames = list(np.unique([item[0] for item in labels_df.columns]))
    if selected_subgroups=="all":
        selected_subgroups=list(config["subgroups"].keys())
    n_groups = len(selected_subgroups)
    # n_groups = len(list(np.unique([item[0] for item in labels_df.columns])))
    fps = config["fps"]
    # start_frame = start * fps
    # stop_frame = stop * fps
    # labels_df = labels_df[start_frame:stop_frame]
    # total_frames = stop_frame - start_frame
    group_dict = {selected_subgroups[i]: i for i in range(len(selected_subgroups))}
    labels_flat = np.array(labels_df)
    labels_flat = [item for sublist in labels_flat for item in sublist]
    clusts = np.unique(labels_flat)
    n_clust = len(clusts)

    label_counts = []
    nbins = int(labels_df.shape[0] / (binsize * config["fps"]))
    for g in range(n_groups):
        for i in range(len(labels_df[selected_subgroups[g]].columns)):
            label_counts_i = np.zeros(n_clust * nbins)
            for b in range(nbins):
                binstart = int(b * (binsize * fps))
                binstop = int((b + 1) * (binsize * fps))
                labels_df_sub = labels_df[binstart:binstop]
                for c in range(n_clust):
                    label_counts_i[c + n_clust * b] = np.count_nonzero(
                        labels_df_sub[selected_subgroups[g]][[labels_df[selected_subgroups[g]].columns[i]]] == c) / (5 * 60 * fps)
            label_counts.append(label_counts_i)
    label_counts = np.array(label_counts)

    group_labels = []
    for g in range(n_groups):
        group_labels.extend([g] * len(labels_df[selected_subgroups[g]].columns))

    lda = LDA(n_components=ncomponents)
    lda_embeddings = lda.fit_transform(label_counts, group_labels)
    return lda, lda_embeddings, label_counts, group_labels, group_dict, nbins

def lr_labels_timebins(config, labels_df, binsize, selected_subgroups="all"):
    """
    Function to compute LDA for data in timebins
    :param config:
    :param labels_df:
    :param binsize:
    :param selected_subgroups:
    :param ncomponents:
    :return:
    """
    # TODO:: build out logistic regression classification function
    # if groupnames == None:
    #     groupnames = list(np.unique([item[0] for item in labels_df.columns]))
    if selected_subgroups=="all":
        selected_subgroups=list(config["subgroups"].keys())
    n_groups = len(selected_subgroups)
    # n_groups = len(list(np.unique([item[0] for item in labels_df.columns])))
    fps = config["fps"]
    # start_frame = start * fps
    # stop_frame = stop * fps
    # labels_df = labels_df[start_frame:stop_frame]
    # total_frames = stop_frame - start_frame
    group_dict = {selected_subgroups[i]: i for i in range(len(selected_subgroups))}
    labels_flat = np.array(labels_df)
    labels_flat = [item for sublist in labels_flat for item in sublist]
    clusts = np.unique(labels_flat)
    n_clust = len(clusts)

    label_counts = []
    nbins = int(labels_df.shape[0] / (binsize * config["fps"]))
    for g in range(n_groups):
        for i in range(len(labels_df[selected_subgroups[g]].columns)):
            label_counts_i = np.zeros(n_clust * nbins)
            for b in range(nbins):
                binstart = int(b * (binsize * fps))
                binstop = int((b + 1) * (binsize * fps))
                labels_df_sub = labels_df[binstart:binstop]
                for c in range(n_clust):
                    label_counts_i[c + n_clust * b] = np.count_nonzero(
                        labels_df_sub[selected_subgroups[g]][[labels_df[selected_subgroups[g]].columns[i]]] == c) / (5 * 60 * fps)
            label_counts.append(label_counts_i)
    label_counts = np.array(label_counts)

    group_labels = []
    for g in range(n_groups):
        group_labels.extend([g] * len(labels_df[selected_subgroups[g]].columns))

    lr = LR().fit(label_counts, group_labels)
    return lr, group_labels, label_counts, group_dict, nbins

def loocv_conf_mat(model, features, group_labels, group_dict):
    loo = LeaveOneOut()
    label_pred = cross_val_predict(model, features, group_labels, cv=loo)
    accuracy = accuracy_score(group_labels, label_pred)
    confusion = confusion_matrix(group_labels, label_pred)

    # Get class labels (group names) in the order of group_dict keys
    class_labels = [key for key, _ in sorted(group_dict.items(), key=lambda item: item[1])]
    class_num = []
    for key in group_dict.keys():
        class_num.append(group_dict[key])
    print("The overall accuracy by leave-one-out-cross-validation (LOOCV) is " + str(accuracy))
    return confusion, class_num, class_labels, accuracy

def nlp_classification(config, labels_df):
    print("Coming soon!")
    # TODO:: add NLP classification function