import numpy as np
import pandas as pd
import os
from sklearn.tree import plot_tree
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression as LR
import scipy

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
    elif config["data_type"]=="MotionMapper":
        label_paths=config["project_files"]
        for i in range(len(label_paths)):
            header.append(label_paths[i])
            labels_i = scipy.io.loadmat(config["data_directory"]+"/"+label_paths[i])["ethogram_data"]
            labels_i = np.sum(labels_i,axis=1)
            if i == 0:
                labels_df = pd.DataFrame(labels_i, columns=[header])
            else:
                labels_df.insert(loc=count, value=labels_i, column=label_paths[i])
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
    elif config["data_type"]=="MotionMapper":
        for g in range(n_groups):
            label_paths_g = config["subgroups"][selected_subgroups[g]]
            for i in range(len(label_paths_g)):
                header = label_paths_g[i]
                header2.append(header)
                header1.append(selected_subgroups[g])
                labels_i = scipy.io.loadmat(config["data_directory"]+"/"+label_paths_g[i])["ethogram_data"]
                labels_i = np.sum(labels_i,axis=1)
                if i == 0 and g == 0:
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

def get_distance_timebins(DLC_config,filepath,binsize,start,end,bodypart,thresh=70):
    """
    Get distance/locomotion for a body part from a DLC file
    :param DLC_config: config file (can be DLC or pose - used only for "fps")
    :param filepath: path to file of interest
    :param binsize: size of timebins (in seconds)
    :param start: start time
    :param end: end time
    :param bodypart: which bodypart to track
    :param thresh: threshold for point-to-point distance that should be marked as wrong and excluded
    :return:
    """
    data=pd.read_csv(filepath,header=[1,2])
    nbins=int((end-start)/binsize)
    dist=np.zeros(nbins)
    for b in range(nbins):
        x = np.array(data[bodypart]['x'])[(start+b*binsize)*DLC_config["fps"]:(start+(b+1)*binsize)*DLC_config["fps"]]
        y = np.array(data[bodypart]['y'])[(start+b*binsize)*DLC_config["fps"]:(start+(b+1)*binsize)*DLC_config["fps"]]
        for i in range(len(x)-1):
            if np.absolute(x[i+1]-x[i])>thresh:
                x[i+1]=x[i]
            if np.absolute(y[i+1]-y[i])>thresh:
                y[i+1]=y[i]
            dist[b]=dist[b]+np.sqrt((x[i+1]-x[i])**2+(y[i+1]-y[i])**2)
    return dist

def dist_df_subgroups(DLC_config, binsize, start, end, thresh=70, selected_subgroups="all"):
    if selected_subgroups=="all":
        selected_subgroups=DLC_config["subgroups"].keys()
    count = 0
    header1 = []
    header2 = []
    for g, group in enumerate(selected_subgroups):
        group_data=[]
        for s, sess in enumerate(DLC_config["subgroups"][group]):
            header = sess
            header2.append(header)
            header1.append(group)
            path = DLC_config["path"]+sess
            dist_i = get_distance_timebins(DLC_config,path,binsize,start,end,"tailbase",thresh=thresh)
            if s == 0 and g == 0:
                dist_df = pd.DataFrame(dist_i, columns=[header])
            else:
                dist_df.insert(loc=count, value=dist_i, column=header)
            count = count + 1
    dist_df.columns = [header1, header2]
    dist_df.index=np.arange(start/60,end/60,binsize/60)
    return dist_df

def BORIS_to_pose(config):
    """
    Intake paired BORIS one-hot-encoded observation files and pose segmented files and align them to see what behaviors
    line up with what pose modules
    :param config: the config
    :return:
    """
    boris_directory = config["boris_directory"]
    boris_to_pose_pairings = config["boris_to_pose_pairings"]
    results=None
    config_modulo = config
    config_modulo["project_files"]=[pairing[1] for pairing in boris_to_pose_pairings if pairing[0] is not None]
    labels_df, n_modules = label_counter_nosubgroups(config_modulo, 0, 1200)
    modules = np.unique(labels_df.values.flatten())
    for pairing in boris_to_pose_pairings:
        if pairing[0]==None:
            continue
        else:
            config_modulo=config
            config_modulo["project_files"]=[pairing[1]]
            labels_df, _ = label_counter_nosubgroups(config_modulo,0,1200)
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
                results = pd.DataFrame(data=np.zeros([n_modules,len(behaviors)]),columns=behaviors,index=modules)
            results_i = pd.DataFrame(columns=behaviors, index=modules)
            for behavior in behaviors:
                mask = boris_i[behavior]>0
                masked_labels = labels_df[mask==True]
                for module in range(n_modules):
                    results_i.at[module,behavior] = np.sum(masked_labels==module).to_numpy()
            results=results+results_i
            print("done with " + str(pairing))
    results=results.T
    column_sums = results.sum()
    column_sums = column_sums.replace(0, 1)
    normalized_results = results / column_sums
    loss_score = np.sum(results.sum()-results.max())/np.sum(results.sum())
    print(config["data_type"] + " modules map onto scored behaviors with 'loss' of " + str(loss_score))
    return results, normalized_results, loss_score

def combine_pose_modules(config, labels_df):
    """
    Combine pose modules based on remappings key in config
    :param config: config
    :param labels_df: from label_counter (subgroups or no_subgroups)
    :return:
    """
    for remapping in config["remappings"]:
        if remapping[0] is not None:
            for old_val in remapping[0]:
                for column in labels_df.columns:
                    labels_df[labels_df == old_val] = remapping[1]
    return labels_df

def make_remappings_from_BORIS(config, labels_df, BORIS_to_pose_mat):
    """
    Make remappings based on BORIS output and apply to labels_df
    :param config: config
    :param labels_df:
    :param BORIS_to_pose_mat: the non-normalized result matrix (first result option) from BORIS_to_pose
    :return:
    """
    BORIS_to_pose_mat_numeric = BORIS_to_pose_mat.apply(pd.to_numeric, errors='coerce')
    new_mappings = list(BORIS_to_pose_mat_numeric.idxmax())
    new_mappings = ['other' if x not in BORIS_to_pose_mat.index else x for x in new_mappings]
    old_mappings =list(BORIS_to_pose_mat_numeric.idxmax().index)
    remappings = [[[old_mappings[i]],new_mappings[i]] for i in range(len(old_mappings))]
    config["remappings"] = remappings
    combine_pose_modules(config, labels_df)
    return labels_df

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