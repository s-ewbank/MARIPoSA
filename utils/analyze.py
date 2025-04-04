import numpy as np
import pandas as pd
import os
import copy
from sklearn.tree import plot_tree
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression as LR
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from scipy import stats
from statsmodels.stats.multitest import multipletests

import scipy
from scipy.spatial.distance import euclidean, mahalanobis, cityblock

# def label_counter_nosubgroups(config, start, stop):
#     """
#     Generates a Pandas dataframe containing the labels for every frame in the specified time range for the video paths in defined groups.
#
#     :param config: the config
#     :param start: time in seconds to start dataframe from.
#     :param stop: time in seconds to stop dataframe at.
#     :param fps: frames per second of recording.
#     :return:
#     """
#     fps = config["fps"]
#     start_frame = start * int(fps)
#     stop_frame = stop * int(fps)
#     count = 0
#     header = []
#     if config["data_source"]=="B-SOiD":
#         label_paths=[i for i in config["project_files"] if "labels_" in i]
#         for i in range(len(label_paths)):
#             header.append(label_paths[i])
#             labels_i = np.loadtxt(
#                 str(config["data_directory"] + "/" + label_paths[i]),
#                 delimiter=",", skiprows=3, usecols=1)[start_frame:stop_frame]
#             if i == 0:
#                 labels_df = pd.DataFrame(labels_i, columns=[header])
#             else:
#                 labels_df = labels_df.copy()
#                 labels_df.insert(loc=count, value=labels_i, column=label_paths[i])
#             count = count + 1
#     elif config["data_source"]=="Keypoint-MoSeq":
#         label_paths=config["project_files"]
#         for i in range(len(label_paths)):
#             header.append(label_paths[i])
#             labels_i = np.loadtxt(
#                 str(config["data_directory"] + "/" + label_paths[i]),
#                 delimiter=",", skiprows=1, usecols=0)[start_frame:stop_frame]
#             if i == 0:
#                 labels_df = pd.DataFrame(labels_i, columns=[header])
#             else:
#                 labels_df.insert(loc=count, value=labels_i, column=label_paths[i])
#                 labels_df = labels_df.copy()
#             count = count + 1
#     elif config["data_source"]=="VAME":
#         data_directory = config["data_directory"]
#         model_path = ""
#         model_path = model_path + "/" + os.listdir(data_directory + "/" + os.listdir(data_directory)[0])[0]
#         model_path = model_path + "/" + os.listdir(data_directory + "/" + os.listdir(data_directory)[0] + model_path)[
#             0] + "/"
#         for f in range(len(config["project_files"])):
#             header.append(config["project_files"][f])
#             datstr = [i for i in os.listdir(data_directory + "/" + config["project_files"][f] + model_path) if "_label_" in i][0]
#             labels_i = np.load(data_directory + "/" + config["project_files"][f] + model_path + datstr)[
#                        start_frame:stop_frame]
#             if f == 0:
#                 labels_df = pd.DataFrame(labels_i, columns=[header])
#             else:
#                 labels_df.insert(loc=count, value=labels_i, column=config["project_files"][f])
#                 labels_df = labels_df.copy()
#             count = count + 1
#     elif config["data_source"]=="MotionMapper":
#         label_paths=config["project_files"]
#         for i in range(len(label_paths)):
#             header.append(label_paths[i])
#             labels_i = scipy.io.loadmat(config["data_directory"]+"/"+label_paths[i])["ethogram_data"]
#             labels_i = np.sum(labels_i,axis=1)
#             if i == 0:
#                 labels_df = pd.DataFrame(labels_i, columns=[header])
#             else:
#                 labels_df.insert(loc=count, value=labels_i, column=label_paths[i])
#                 labels_df = labels_df.copy()
#             count = count + 1
#
#     #Get number of modules
#     labels_flat = np.array(labels_df)
#     labels_flat = [item for sublist in labels_flat for item in sublist]
#     modules = np.unique(labels_flat)
#     n_modules = len(modules)
#     return labels_df
#
# def label_counter_subgroups(config, start, stop, selected_subgroups="all"):
#     """
#     Generates a Pandas dataframe containing the labels for every frame in the specified time range for the video paths in defined groups.
#
#     :param config: the config
#     :param start: time in seconds to start dataframe from.
#     :param stop: time in seconds to stop dataframe at.
#     :param fps: frames per second of recording.
#     :param selected_subgroups: frames per second of recording.
#     :return:
#     """
#     fps = config["fps"]
#     start_frame = start * int(fps)
#     stop_frame = stop * int(fps)
#     if selected_subgroups=="all":
#         selected_subgroups=list(config["subgroups"].keys())
#     n_groups = len(selected_subgroups)
#     count = 0
#     header1 = []
#     header2 = []
#     if config["data_source"]=="B-SOiD":
#         for g in range(n_groups):
#             label_paths_g = [i for i in config["subgroups"][selected_subgroups[g]] if "labels_" in i]
#             for i in range(len(label_paths_g)):
#                 header = label_paths_g[i]
#                 header2.append(header)
#                 header1.append(selected_subgroups[g])
#                 labels_i = np.loadtxt(
#                     str(config["data_directory"] + "/" + label_paths_g[i]),
#                     delimiter=",", skiprows=3, usecols=1)[start_frame:stop_frame]
#                 if i == 0 and g == 0:
#                     labels_df = pd.DataFrame(labels_i, columns=[header])
#                 else:
#                     labels_df.insert(loc=count, value=labels_i, column=header)
#                     labels_df = labels_df.copy()
#                 count = count + 1
#         labels_df.columns = [header1, header2]
#     elif config["data_source"]=="Keypoint-MoSeq":
#         for g in range(n_groups):
#             label_paths_g = config["subgroups"][selected_subgroups[g]]
#             for i in range(len(label_paths_g)):
#                 header = label_paths_g[i]
#                 header2.append(header)
#                 header1.append(selected_subgroups[g])
#                 labels_i = np.loadtxt(
#                     str(config["data_directory"] + "/" + label_paths_g[i]),
#                     delimiter=",", skiprows=1, usecols=0)[start_frame:stop_frame]
#                 if i == 0 and g == 0:
#                     labels_df = pd.DataFrame(labels_i, columns=[header])
#                 else:
#                     labels_df.insert(loc=count, value=labels_i, column=header)
#                     labels_df = labels_df.copy()
#                 count = count + 1
#         labels_df.columns = [header1, header2]
#     elif config["data_source"]=="VAME":
#         data_directory = config["data_directory"]
#         model_path = ""
#         model_path = model_path + "/" + os.listdir(data_directory + "/" + os.listdir(data_directory)[0])[0]
#         model_path = model_path + "/" + os.listdir(data_directory + "/" + os.listdir(data_directory)[0] + model_path)[
#             0] + "/"
#         for g in range(n_groups):
#             label_paths_g = config["subgroups"][selected_subgroups[g]]
#             for f in range(len(label_paths_g)):
#                 header = label_paths_g[f]
#                 header2.append(header)
#                 header1.append(selected_subgroups[g])
#                 datstr = [i for i in os.listdir(data_directory + "/" + label_paths_g[f] + model_path) if "_label_" in i][0]
#                 labels_i = np.load(data_directory + "/" + label_paths_g[f] + model_path + datstr)[
#                            start_frame:stop_frame]
#                 if f == 0 and g == 0:
#                     labels_df = pd.DataFrame(labels_i, columns=[header])
#                 else:
#                     labels_df.insert(loc=count, value=labels_i, column=header)
#                     labels_df = labels_df.copy()
#                 count = count + 1
#         labels_df.columns = [header1, header2]
#     elif config["data_source"]=="MotionMapper":
#         for g in range(n_groups):
#             label_paths_g = config["subgroups"][selected_subgroups[g]]
#             for i in range(len(label_paths_g)):
#                 header = label_paths_g[i]
#                 header2.append(header)
#                 header1.append(selected_subgroups[g])
#                 labels_i = scipy.io.loadmat(config["data_directory"]+"/"+label_paths_g[i])["ethogram_data"]
#                 labels_i = np.sum(labels_i,axis=1)
#                 if i == 0 and g == 0:
#                     labels_df = pd.DataFrame(labels_i, columns=[header])
#                 else:
#                     labels_df.insert(loc=count, value=labels_i, column=header)
#                 count = count + 1
#         labels_df.columns = [header1, header2]
#
#     #Get number of modules
#     labels_flat = np.array(labels_df)
#     labels_flat = [item for sublist in labels_flat for item in sublist]
#     modules = np.unique(labels_flat)
#     n_modules = len(modules)
#     return labels_df


def is_nonnum(value):
    try:
        float(value)
        return False
    except (ValueError, TypeError):
        return True

def packed_moving_average(x, w):
    x = np.asarray(x)
    n = len(x)
    x_new = np.zeros(n)

    for i in range(n):
        win_size = min(w, i + 1, n - i)  # Reduce window size at edges
        x_new[i] = np.mean(x[max(0, i - win_size // 2): min(n, i + win_size // 2 + 1)])

    return x_new

def get_module_labels(config, start, stop, subgroups = None):
    """
    Generates a Pandas dataframe containing the labels for every frame in the specified time range for the video paths in defined groups.

    :param config: the config
    :param start: time in seconds to start dataframe from.
    :param stop: time in seconds to stop dataframe at.
    :param fps: frames per second of recording.
    :param subgroups: subgroups to include; by default, None will result in an object without data subgrouped; could alternatively be a list of subgroup names from config or "all" (to include all subgroups present in config)
    :return:
    """
    fps = config["fps"]
    start_frame = start * int(fps)
    stop_frame = stop * int(fps)
    if subgroups==None:
        count = 0
        header = []
        if config["data_source"]=="B-SOiD":
            label_paths=[i for i in config["project_files"] if "labels_" in i]
            for i in range(len(label_paths)):
                header.append(label_paths[i])
                labels_i = np.loadtxt(
                    str(config["data_directory"] + "/" + label_paths[i]),
                    delimiter=",", skiprows=3, usecols=1)[start_frame:stop_frame]
                if i == 0:
                    labels_df = pd.DataFrame(labels_i, columns=[header])
                else:
                    labels_df = labels_df.copy()
                    labels_df.insert(loc=count, value=labels_i, column=label_paths[i])
                count = count + 1
        elif config["data_source"]=="Keypoint-MoSeq":
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
                    labels_df = labels_df.copy()
                count = count + 1
        elif config["data_source"]=="VAME":
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
                    labels_df = labels_df.copy()
                count = count + 1
        elif config["data_source"]=="MotionMapper":
            label_paths=config["project_files"]
            for i in range(len(label_paths)):
                header.append(label_paths[i])
                labels_i = scipy.io.loadmat(config["data_directory"]+"/"+label_paths[i])["ethogram_data"]
                labels_i = np.sum(labels_i,axis=1)
                if i == 0:
                    labels_df = pd.DataFrame(labels_i, columns=[header])
                else:
                    labels_df.insert(loc=count, value=labels_i, column=label_paths[i])
                    labels_df = labels_df.copy()
                count = count + 1

    else:
        if subgroups=="all":
            subgroups=list(config["subgroups"].keys())
        n_groups = len(subgroups)
        count = 0
        header1 = []
        header2 = []
        if config["data_source"]=="B-SOiD":
            for g in range(n_groups):
                label_paths_g = [i for i in config["subgroups"][subgroups[g]] if "labels_" in i]
                for i in range(len(label_paths_g)):
                    header = label_paths_g[i]
                    header2.append(header)
                    header1.append(subgroups[g])
                    labels_i = np.loadtxt(
                        str(config["data_directory"] + "/" + label_paths_g[i]),
                        delimiter=",", skiprows=3, usecols=1)[start_frame:stop_frame]
                    if i == 0 and g == 0:
                        labels_df = pd.DataFrame(labels_i, columns=[header])
                    else:
                        labels_df.insert(loc=count, value=labels_i, column=header)
                        labels_df = labels_df.copy()
                    count = count + 1
            labels_df.columns = [header1, header2]
        elif config["data_source"]=="Keypoint-MoSeq":
            for g in range(n_groups):
                label_paths_g = config["subgroups"][subgroups[g]]
                for i in range(len(label_paths_g)):
                    header = label_paths_g[i]
                    header2.append(header)
                    header1.append(subgroups[g])
                    labels_i = np.loadtxt(
                        str(config["data_directory"] + "/" + label_paths_g[i]),
                        delimiter=",", skiprows=1, usecols=0)[start_frame:stop_frame]
                    if i == 0 and g == 0:
                        labels_df = pd.DataFrame(labels_i, columns=[header])
                    else:
                        labels_df.insert(loc=count, value=labels_i, column=header)
                        labels_df = labels_df.copy()
                    count = count + 1
            labels_df.columns = [header1, header2]
        elif config["data_source"]=="VAME":
            data_directory = config["data_directory"]
            model_path = ""
            model_path = model_path + "/" + os.listdir(data_directory + "/" + os.listdir(data_directory)[0])[0]
            model_path = model_path + "/" + os.listdir(data_directory + "/" + os.listdir(data_directory)[0] + model_path)[
                0] + "/"
            for g in range(n_groups):
                label_paths_g = config["subgroups"][subgroups[g]]
                for f in range(len(label_paths_g)):
                    header = label_paths_g[f]
                    header2.append(header)
                    header1.append(subgroups[g])
                    datstr = [i for i in os.listdir(data_directory + "/" + label_paths_g[f] + model_path) if "_label_" in i][0]
                    labels_i = np.load(data_directory + "/" + label_paths_g[f] + model_path + datstr)[
                               start_frame:stop_frame]
                    if f == 0 and g == 0:
                        labels_df = pd.DataFrame(labels_i, columns=[header])
                    else:
                        labels_df.insert(loc=count, value=labels_i, column=header)
                        labels_df = labels_df.copy()
                    count = count + 1
            labels_df.columns = [header1, header2]
        elif config["data_source"]=="MotionMapper":
            for g in range(n_groups):
                label_paths_g = config["subgroups"][subgroups[g]]
                for i in range(len(label_paths_g)):
                    header = label_paths_g[i]
                    header2.append(header)
                    header1.append(subgroups[g])
                    labels_i = scipy.io.loadmat(config["data_directory"]+"/"+label_paths_g[i])["ethogram_data"]
                    labels_i = np.sum(labels_i,axis=1)
                    if i == 0 and g == 0:
                        labels_df = pd.DataFrame(labels_i, columns=[header])
                    else:
                        labels_df.insert(loc=count, value=labels_i, column=header)
                    count = count + 1
            labels_df.columns = [header1, header2]
    return labels_df

def get_distance_timebins(PE_config,filepath,binsize,start,end,bodypart,thresh=70):
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
        x = np.array(data[bodypart]['x'])[(start+b*binsize)*PE_config["fps"]:(start+(b+1)*binsize)*PE_config["fps"]]
        y = np.array(data[bodypart]['y'])[(start+b*binsize)*PE_config["fps"]:(start+(b+1)*binsize)*PE_config["fps"]]
        for i in range(len(x)-1):
            if np.absolute(x[i+1]-x[i])>thresh:
                x[i+1]=x[i]
            if np.absolute(y[i+1]-y[i])>thresh:
                y[i+1]=y[i]
            dist[b]=dist[b]+np.sqrt((x[i+1]-x[i])**2+(y[i+1]-y[i])**2)
    return dist

def dist_df_subgroups(PE_config, binsize, start, end, thresh=70, selected_subgroups="all"):
    if selected_subgroups=="all":
        selected_subgroups=PE_config["subgroups"].keys()
    count = 0
    header1 = []
    header2 = []
    for g, group in enumerate(selected_subgroups):
        group_data=[]
        for s, sess in enumerate(PE_config["subgroups"][group]):
            header = sess
            header2.append(header)
            header1.append(group)
            path = PE_config["path"]+sess
            dist_i = get_distance_timebins(PE_config,path,binsize,start,end,"tailbase",thresh=thresh)
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
    # TODO: Make possible inputs include aggregate table
    boris_directory = config["boris_directory"]
    boris_to_pose_pairings = config["boris_to_pose_pairings"]
    results=None
    config_modulo = copy.deepcopy(config)
    config_modulo["project_files"]=[pairing[1] for pairing in boris_to_pose_pairings if pairing[0] is not None]
    labels_df = get_module_labels(config_modulo, 0, 1200)
    labels_flat = np.array(labels_df)
    labels_flat = [item for sublist in labels_flat for item in sublist]
    modules = np.unique(labels_flat)
    n_modules = len(modules)
    modules = np.unique(labels_df.values.flatten())
    for pairing in boris_to_pose_pairings:
        if pairing[0]==None:
            continue
        else:
            config_modulo = copy.deepcopy(config)
            config_modulo["project_files"]=[pairing[1]]
            labels_df = get_module_labels(config_modulo,0,1200)
            boris_i = pd.read_csv(boris_directory + "/" + pairing[0])
            boris_i["frame"] = np.round(boris_i["time"] / (1 / int(config["fps"])))
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
                    results_i.at[module,behavior] = np.sum(masked_labels==module).to_numpy()[0]
            results=results+results_i
            print("done with " + str(pairing))
    results=results.T
    column_sums = results.sum()
    column_sums = column_sums.replace(0, 1)
    normalized_results = results / column_sums
    loss_score = np.sum(results.sum()-results.max())/np.sum(results.sum())
    print(config["data_source"] + " modules map onto scored behaviors with 'loss' of " + str(loss_score))
    return results, normalized_results, loss_score

def combine_pose_modules(config, labels_df, force_no_numeric=False):
    """
    Combine pose modules based on remappings key in config

    :param config: config
    :param labels_df: from label_counter (subgroups or no_subgroups)
    :return:
    """
    n_rules=len(config["remappings"])
    print("Applying " + str(n_rules) + " remapping rules from config.")
    for r, remapping in enumerate(config["remappings"]):
        print(str(r+1) + " of " + str(n_rules) + ": Remapping " + str(remapping[0]) + " to " + str(remapping[1]))
        if remapping[0] is not None:
            for old_val in remapping[0]:
                for column in labels_df.columns:
                    labels_df.replace(old_val, remapping[1], inplace=True)
    if force_no_numeric:
        labels_flat = np.array(labels_df)
        labels_flat = [item for sublist in labels_flat for item in sublist]
        modules = np.unique(labels_flat)
        nonnum_vals = [is_nonnum(i)==False for i in modules]
        if np.sum(nonnum_vals)>0:
            print("Warning - numeric values detected and force_no_numeric set to True. The following modules will be changed to 'other':")
            print(modules[nonnum_vals])
            for module in modules[nonnum_vals]:
                for column in labels_df.columns:
                    labels_df.replace(module, "other", inplace=True)
                    try:
                        labels_df.replace(int(module), "other", inplace=True)
                    except:
                        pass
                    try:
                        labels_df.replace(float(module), "other", inplace=True)
                    except:
                        pass
    return labels_df


def make_remappings_from_BORIS(config, labels_df=None, BORIS_to_pose_mat=None, force_no_numeric=False):
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
    if labels_df is not None:
        combine_pose_modules(config, labels_df, force_no_numeric=force_no_numeric)
        return labels_df
    else:
        return config

# class UsageFeats:
#     def __init__(self, label_counts, group_labels, feat_names, group_dict):
#         self.label_counts = label_counts
#         self.group_labels = group_labels
#         self.feat_names = feat_names
#         self.group_dict = group_dict
#         mean_check = np.allclose(np.mean(label_counts, axis=0), 0, atol=0.1)
#         std_check = np.allclose(np.std(label_counts, axis=0), 1, atol=0.1)
#         self.scaled = mean_check and std_check
#
#     def to_df(self):
#         colnames=[]
#         flip_group_dict = {v: k for k, v in self.group_dict.items()}
#         for i in self.group_labels:
#             colnames.append(flip_group_dict[i])
#         df = pd.DataFrame(self.label_counts, columns=self.feat_names, index=colnames)
#         return df
#
#     def collapse_timebins(self):
#         colnames=[]
#         flip_group_dict = {v: k for k, v in self.group_dict.items()}
#         for i in self.group_labels:
#             colnames.append(flip_group_dict[i])
#         df = pd.DataFrame(self.label_counts, columns=self.feat_names, index=colnames)
#         extracted = [i.split("_")[0] for i in df.columns]
#         modules = pd.Series(extracted).unique()
#
#         df_notimebins = pd.DataFrame(index=df.index, columns=modules)
#         for module in modules:
#             df_notimebins[module] = df.filter(like=module + "_").mean(axis=1)
#
#         return UsageFeats(np.array(df_notimebins), self.group_labels, df_notimebins.columns, self.group_dict)
#
#     def apply_picks(self,pick_names):
#         feats = self.to_df().columns
#         if set(pick_names)<=set(feats):
#             colnames=[]
#             flip_group_dict = {v: k for k, v in self.group_dict.items()}
#             for i in self.group_labels:
#                 colnames.append(flip_group_dict[i])
#             df = pd.DataFrame(self.label_counts, columns=self.feat_names, index=colnames)
#             df_sub = df[pick_names]
#             return UsageFeats(np.array(df_sub), self.group_labels, df_sub.columns, self.group_dict)
#         else:
#             print("Picks not found in features, presuming that you are applying no-bin picks to binned data and adjusting accordingly...")
#             print("Input picks: {}".format(pick_names))
#             new_picks = []
#             for pick in pick_names:
#                 new_picks.extend([i for i in feats if pick+"_" in i])
#             print("Applied picks: {}".format(new_picks))
#             picked_feats = self.apply_picks(new_picks)
#             return picked_feats
#
#     def scale(self):
#         scaler = StandardScaler()
#         label_counts_scaled = scaler.fit_transform(self.label_counts)
#         return UsageFeats(label_counts_scaled, self.group_labels, self.feat_names, self.group_dict)
#
#
# def get_usage_feats(config,
#                     labels_df,
#                     binsize,
#                     selected_subgroups="all"):
#     """
#     Reshape labels dataframe from label_counter_subgroups to be an array of features
#
#     :param config: config object
#     :param labels_df: labels dataframe from label_counter_subgroups
#     :param binsize: width of bins in seconds
#     :param selected_subgroups:
#     :return:
#     """
#     if selected_subgroups=="all":
#         selected_subgroups=list(config["subgroups"].keys())
#     n_groups = len(selected_subgroups)
#     fps = int(config["fps"])
#     group_dict = {selected_subgroups[i]: i for i in range(len(selected_subgroups))}
#     labels_flat = np.array(labels_df)
#     labels_flat = [item for sublist in labels_flat for item in sublist]
#     clusts = np.unique(labels_flat)
#     n_clust = len(clusts)
#
#     label_counts = []
#
#     nbins = int(labels_df.shape[0] / (binsize * fps))
#     feat_names_made=False
#     feat_names=[]
#
#     group_labels = []
#     for g in range(n_groups):
#         for i in range(len(labels_df[selected_subgroups[g]].columns)):
#             label_counts_i = np.zeros(n_clust * nbins)
#             for b in range(nbins):
#                 binstart = int(b * (binsize * fps))
#                 binstop = int((b + 1) * (binsize * fps))
#                 labels_df_sub = labels_df[binstart:binstop]
#                 for c in range(n_clust):
#                     label_counts_i[c + n_clust * b] = np.count_nonzero(
#                         labels_df_sub[selected_subgroups[g]][[labels_df_sub[selected_subgroups[g]].columns[i]]] == c) / (binsize * fps)
#                     if feat_names_made==False:
#                         if nbins>1:
#                             feat_names.append(f"module{c}_t{int(binstart/fps)}-{int(binstop/fps)}")
#                         else:
#                             feat_names.append(f"module{c}")
#             label_counts.append(label_counts_i)
#             group_labels.append(g)
#             feat_names_made=True
#     label_counts = np.array(label_counts)
#
#     return UsageFeats(label_counts, group_labels, feat_names, group_dict)

class ModuleUsage:
    def __init__(self, label_counts, group_labels, observation_labels, feat_names, group_dict, scaler):
        self.label_counts = label_counts
        self.group_labels = group_labels
        self.observation_labels = observation_labels
        self.feat_names = feat_names
        self.group_dict = group_dict
        self.scaler = scaler
        mean_check = np.allclose(np.mean(label_counts, axis=0), 0, atol=0.1)
        std_check = np.allclose(np.std(label_counts, axis=0), 1, atol=0.1)
        self.scaled = mean_check and std_check

    def to_df(self):
        colnames = []
        flip_group_dict = {v: k for k, v in self.group_dict.items()}
        bad_group_labels = []
        for i in self.group_labels:
            if i in flip_group_dict.keys():
                colnames.append(flip_group_dict[i])
            else:
                colnames.append("none")
                bad_group_labels.append(i)
        if len(bad_group_labels) > 0:
            bad_group_labels = list(np.unique(bad_group_labels))
            print(f"Warning: the following groups were not detected in the config and were labeled 'none'; continuing:\n{bad_group_labels}")
        df = pd.DataFrame(self.label_counts, columns=self.feat_names, index=self.observation_labels)
        df["group"] = colnames
        return df

    def collapse_timebins(self):
        colnames = []
        flip_group_dict = {v: k for k, v in self.group_dict.items()}
        for i in self.group_labels:
            colnames.append(flip_group_dict[i])
        df = pd.DataFrame(self.label_counts, columns=self.feat_names, index=colnames)
        extracted = [i.split("_")[0] for i in df.columns]
        modules = pd.Series(extracted).unique()

        df_notimebins = pd.DataFrame(index=df.index, columns=modules)
        for module in modules:
            df_notimebins[module] = df.filter(like=module + "_").mean(axis=1)

        return ModuleUsage(np.array(df_notimebins), self.group_labels, self.observation_labels, df_notimebins.columns,
                           self.group_dict, self.scaler)

    def usage_density(self, convolve=False, window=5):
        df = self.to_df()
        modules = sorted(
            np.unique([int(i.split("module")[1].split("_")[0]) for i in list(df.columns) if "module" in i]))
        start_times = [int(col.split("_t")[1].split("-")[0]) for col in list(df.columns) if
                       f"module{modules[0]}_" in col]
        subjs = list(df.index)
        data = []
        for subj in subjs:
            new_df = pd.DataFrame(columns=start_times, index=modules)
            for module in modules:
                subj_module_arr = np.array(df.filter(like=f"module{module}_", axis=1).loc[subj])
                if convolve:
                    subj_module_arr = packed_moving_average(subj_module_arr, window)
                new_df.loc[module] = subj_module_arr
            data.append(new_df)

        return data

    def apply_picks(self, pick_names):
        feats = self.to_df().columns
        if set(pick_names) <= set(feats):
            colnames = []
            flip_group_dict = {v: k for k, v in self.group_dict.items()}
            for i in self.group_labels:
                colnames.append(flip_group_dict[i])
            df = pd.DataFrame(self.label_counts, columns=self.feat_names, index=colnames)
            df_sub = df[pick_names]
            return ModuleUsage(np.array(df_sub), self.group_labels, self.observation_labels, df_sub.columns,
                               self.group_dict, self.scaler)
        else:
            print(
                "Picks not found in features, presuming that you are applying no-bin picks to binned data and adjusting accordingly...")
            print("Input picks: {}".format(pick_names))
            new_picks = []
            for pick in pick_names:
                new_picks.extend([i for i in feats if pick + "_" in i])
            print("Applied picks: {}".format(new_picks))
            picked_feats = self.apply_picks(new_picks)
            return picked_feats

    def scale(self):
        scaler = StandardScaler()
        label_counts_scaled = scaler.fit_transform(self.label_counts)
        return ModuleUsage(label_counts_scaled, self.group_labels, self.observation_labels, self.feat_names,
                           self.group_dict, scaler)

    def f_oneway(self):
        df = self.to_df()
        all_mod_usage = {}
        module_names = [i for i in list(df.columns) if i != "group"]
        classes = list(df[("group")].unique())

        results_df = []
        for m in module_names:
            mod_m_usage = []
            for class_c in classes:
                mod_m_usage.append(df[df["group"] == class_c][m].to_numpy())
            all_mod_usage[m] = mod_m_usage
            result = stats.f_oneway(*all_mod_usage[m])
            results_df.append({"module": m, "f": result.statistic, "p_uncorr": result.pvalue})
        results_df = pd.DataFrame(results_df)
        p_uncorr = np.array(results_df["p_uncorr"])
        results_df["p_corr"] = multipletests(p_uncorr, method="fdr_bh")[1]
        return results_df


def get_module_usage(config, labels_df, binsize=None):
    """
    Reshape labels dataframe from label_counter_subgroups to be an array of features

    :param config: config object
    :param labels_df: labels dataframe from label_counter_subgroups
    :param binsize: width of bins in seconds; if None, no binning is performed
    :param selected_subgroups:
    :return:
    """
    data_subgrouped = False
    try:
        list(labels_df.columns.get_level_values(1).unique())
        subgroups = list(labels_df.columns.get_level_values(0).unique())
        group_dict = {subgroups[i]: i for i in range(len(subgroups))}
        data_subgrouped = True
    except IndexError:
        subgroups = list(labels_df.columns.get_level_values(0).unique())
        group_dict = {"no_assigned_subgroup": 0}
        data_subgrouped = False

    n_groups = len(subgroups)
    fps = int(config["fps"])
    labels_flat = np.array(labels_df)
    labels_flat = [item for sublist in labels_flat for item in sublist]
    modules = np.unique(labels_flat)
    n_modules = len(modules)

    label_counts = []

    if binsize == None:
        binsize = (labels_df.index.stop - labels_df.index.start) / int(config["fps"])

    nbins = int(labels_df.shape[0] / (binsize * fps))
    feat_names_made = False
    feat_names = []

    group_labels = []
    observation_labels = []

    for g in range(n_groups):
        for i in range(len(labels_df[subgroups[g]].columns)):
            label_counts_i = np.zeros(n_modules * nbins)
            for b in range(nbins):
                binstart = int(b * (binsize * fps))
                binstop = int((b + 1) * (binsize * fps))
                labels_df_sub = labels_df[binstart:binstop]
                for m,mod in enumerate(modules):
                    label_counts_i[m + n_modules * b] = np.count_nonzero(
                        labels_df_sub[subgroups[g]][[labels_df_sub[subgroups[g]].columns[i]]] == mod) / (binsize * fps)
                    if feat_names_made == False:
                        if is_nonnum(mod):
                            modname=mod
                        else:
                            modname=str(int(mod))
                        if nbins > 1:
                            feat_names.append(f"module{modname}_t{int(binstart / fps)}-{int(binstop / fps)}")
                        else:
                            feat_names.append(f"module{modname}")
            label_counts.append(label_counts_i)
            if data_subgrouped:
                group_labels.append(g)
                observation_labels.append(labels_df[subgroups[g]].columns[i])
            else:
                group_labels.append(0)
                observation_labels.append(labels_df[subgroups[g]].columns[i][0])
            feat_names_made = True
    label_counts = np.array(label_counts)

    return ModuleUsage(label_counts, group_labels, observation_labels, feat_names, group_dict, None)

class ModuleTransitions:
    def __init__(self, transition_counts, transition_count_matrices, group_labels, observation_labels, feat_names, group_dict):
        self.transition_counts = transition_counts
        self.transition_count_matrices = transition_count_matrices
        self.group_labels = group_labels
        self.observation_labels = observation_labels
        self.feat_names = feat_names
        self.group_dict = group_dict
        mean_check = np.allclose(np.mean(transition_counts, axis=0), 0, atol=0.1)
        std_check = np.allclose(np.std(transition_counts, axis=0), 1, atol=0.1)
        self.scaled = mean_check and std_check

    def to_df(self):
        colnames = []
        flip_group_dict = {v: k for k, v in self.group_dict.items()}
        bad_group_labels = []
        for i in self.group_labels:
            if i in flip_group_dict.keys():
                colnames.append(flip_group_dict[i])
            else:
                colnames.append("none")
                bad_group_labels.append(i)
        if len(bad_group_labels) > 0:
            print(f"Warning: the following groups were not detected in the config and were labeled 'none'; continuing:\n{bad_group_labels}")
        df = pd.DataFrame(self.transition_counts, columns=self.feat_names, index=self.observation_labels)
        df["group"] = colnames
        return df

    def scale(self):
        scaler = StandardScaler()
        transition_counts_scaled = scaler.fit_transform(self.transition_counts)
        #transition_count_matrices_scaled = scaler.fit_transform(self.transition_count_matrices)
        return ModuleTransitions(transition_counts_scaled, self.transition_count_matrices, self.group_labels, self.observation_labels, self.feat_names,
                           self.group_dict)

def get_module_transitions(config, labels_df):
    """
    Reshape labels dataframe from label_counter_subgroups to be an array of features

    :param config: config object
    :param labels_df: labels dataframe from label_counter_subgroups
    :param binsize: width of bins in seconds; if None, no binning is performed
    :param selected_subgroups:
    :return:
    """
    data_subgrouped = False
    try:
        list(labels_df.columns.get_level_values(1).unique())
        subgroups = list(labels_df.columns.get_level_values(0).unique())
        group_dict = {subgroups[i]: i for i in range(len(subgroups))}
        data_subgrouped = True
    except IndexError:
        subgroups = list(labels_df.columns.get_level_values(0).unique())
        group_dict = {"no_assigned_subgroup": 0}
        data_subgrouped = False

    n_groups = len(subgroups)
    fps = int(config["fps"])
    labels_flat = np.array(labels_df)
    labels_flat = [item for sublist in labels_flat for item in sublist]
    modules = np.unique(labels_flat)
    n_modules = len(modules)

    transition_counts = []
    transition_count_matrices = []

    feat_names_made = False
    feat_names = []

    group_labels = []
    observation_labels = []

    for g in range(n_groups):
        for i in range(len(labels_df[subgroups[g]].columns)):
            transition_counts_i = np.zeros(n_modules * n_modules)
            transition_counts_i_matrix = np.zeros([n_modules, n_modules])
            comparison_idx=0
            for m_i, mod_i in enumerate(modules):
                for m_j, mod_j in enumerate(modules):
                    if feat_names_made == False:
                        if is_nonnum(mod_i):
                            modname=f'{mod_i}_to_{mod_j}'
                        else:
                            modname=f'{str(int(mod_i))}_to_{str(int(mod_j))}'
                        feat_names.append(f"{modname}")
                    arr = np.array(labels_df[subgroups[g]][labels_df[subgroups[g]].columns[i]])
                    value=np.sum((arr[:-1] == mod_i) & (arr[1:] == mod_j))
                    transition_counts_i[comparison_idx] = value
                    transition_counts_i_matrix[m_i,m_j] = value
                    comparison_idx += 1
            transition_counts.append(transition_counts_i)
            transition_count_matrices.append(transition_counts_i_matrix)
            if data_subgrouped:
                group_labels.append(g)
                observation_labels.append(labels_df[subgroups[g]].columns[i])
            else:
                group_labels.append(0)
                observation_labels.append(labels_df[subgroups[g]].columns[i][0])
            feat_names_made = True
    transition_counts = np.array(transition_counts)

    return ModuleTransitions(transition_counts, transition_count_matrices, group_labels, observation_labels, feat_names, group_dict)

def embed(module_feature_object,method="lda",n_components=2):
    """
    Get dimensionally reduced space embeddings of the data with LDA or PCA
    :param module_feature_object: module feature object
    :param method: LDA or PCA; default LDA
    :param n_components: number of components
    :return:
    """
    if module_feature_object.__class__.__name__=="ModuleUsage":
        X=module_feature_object.label_counts
        y=module_feature_object.group_labels
        obj_type="module usage"
    elif module_feature_object.__class__.__name__=="ModuleTransitions":
        X=module_feature_object.transition_counts
        y=module_feature_object.group_labels
        obj_type="module transitions"
    else:
        raise ValueError(f'module_feature_object class must be ModuleUsage or ModuleTransitions, not {module_feature_object.__class__}')

    if ((method == "lda") or (method == "LDA") or (method == "lineardiscriminantanalysis") or (method == "LinearDiscriminantAnalysis")):
        emb = LDA(n_components=n_components,store_covariance=True)
        emb.fit(X,y)
    elif ((method == "pca") or (method == "PCA")):
        emb = PCA(n_components=n_components)
        emb.fit(X)
    return emb

def classify(module_feature_object, method="lda"):
    """
    Classify pose segmentation data using either module usage or transitions
    :param module_feature_object: ModuleUsage or ModuleTransitions object
    :param method: classification method to use; options include "lda", "logisticregression", "mlp", "naivebayes", "knn", or "randomforest"
    :return: classifier

    """
    if module_feature_object.__class__.__name__=="ModuleUsage":
        X = module_feature_object.label_counts
        y = module_feature_object.group_labels
        obj_type = "module usage"
    elif module_feature_object.__class__.__name__ == "ModuleTransitions":
        X = module_feature_object.transition_counts
        y = module_feature_object.group_labels
        obj_type = "module transitions"
    else:
        raise ValueError(
            f'module_feature_object class must be ModuleUsage or ModuleTransitions, not {module_feature_object.__class__}')

    if ((method == "lda") or (method == "LDA") or (method == "lineardiscriminantanalysis") or (
            method == "LinearDiscriminantAnalysis")):
        clf = LDA()
    elif ((method == "lr") or (method == "LR") or (method == "LogisticRegression") or (
            method == "logisticregression")):
        clf = LR()
    elif ((method == "mlp") or (method == "MLP") or (method == "nn") or (method == "NN")):
        clf = MLPClassifier(max_iter=1000)
    elif ((method == "naivebayes") or (method == "NaiveBayes") or (method == "nb") or (method == "NB")):
        clf = GaussianNB()
    elif ((method == "k") or (method == "knn") or (method == "KNN")):
        clf = KNeighborsClassifier()
    elif ((method == "rf") or (method == "randomforest") or (method == "RF") or (method == "RandomForest")):
        clf = RandomForestClassifier()
    else:
        raise ValueError(
            f'Invalid method "{method}"; must be one of: lda, lr, mlp, nb, knn, or rf (or one of various alternative spellings)')
    clf.fit(X, y)
    return clf

def loocv(module_feature_object, method="lda"):
    """
    Perform leave-one-out cross-validation for a method of classifying pose segmentation data using either module usage or transitions
    :param module_feature_object: ModuleUsage or ModuleTransitions object
    :param method: classification method to use; options include "lda", "logisticregression", "mlp", "naivebayes", "knn", or "randomforest"
    :return: accuracy, conf_mat

    """
    loocv_preds = []
    if module_feature_object.__class__.__name__=="ModuleUsage":
        X = module_feature_object.label_counts
        y = module_feature_object.group_labels
    elif module_feature_object.__class__.__name__ == "ModuleTransitions":
        X = module_feature_object.transition_counts
        y = module_feature_object.group_labels
    else:
        raise ValueError(
            f'module_feature_object class must be ModuleUsage or ModuleTransitions, not {module_feature_object.__class__}')
    for obs in range(len(y)):
        sub_X = np.delete(X, obs, axis=0)
        sub_y = y[:obs] + y[obs + 1:]
        X_i = X[obs, :]
        y_i = y[obs]
        if ((method == "lda") or (method == "LDA") or (method == "lineardiscriminantanalysis") or (
                method == "LinearDiscriminantAnalysis")):
            clf = LDA()
        elif ((method == "lr") or (method == "LR") or (method == "LogisticRegression") or (
                method == "logisticregression")):
            clf = LR()
        elif ((method == "mlp") or (method == "MLP") or (method == "nn") or (method == "NN")):
            clf = MLPClassifier(max_iter=1000)
        elif ((method == "naivebayes") or (method == "NaiveBayes") or (method == "nb") or (method == "NB")):
            clf = GaussianNB()
        elif ((method == "k") or (method == "knn") or (method == "KNN")):
            clf = KNeighborsClassifier()
        elif ((method == "rf") or (method == "randomforest") or (method == "RF") or (method == "RandomForest")):
            clf = RandomForestClassifier()
        else:
            raise ValueError(
                f'Invalid method "{method}"; must be one of: lda, lr, mlp, nb, knn, or rf (or one of various alternative spellings)')
        clf.fit(sub_X, sub_y)
        loocv_preds.append(clf.predict(X_i.reshape(1, -1))[0])
    loocv_preds = np.array(loocv_preds)
    y = np.array(y)
    accuracy = np.mean(loocv_preds == y)
    classes = sorted(np.unique(y))
    conf_mat = np.zeros([len(classes), len(classes)])
    for i in range(len(classes)):
        for j in range(len(classes)):
            conf_mat[i, j] = np.sum(loocv_preds[y == i] == j)
    return accuracy, conf_mat

def get_distance(module_feature_object, metric="euclidean", method="centroid", embedding=None, hold_out=False,
                 n_components=2):
    """
    Get distance in some method between
    :param module_feature_object: ModuleUsage or ModuleTransitions object
    :param metric: distance metric to use; options are "euclidean","cityblock","mahalanobis"
    :param method: what distance to calculate; options are "pairwise" for between every pair of points or "centroid" for between points to group centroids
    :param hold_out: whether to hold out each sample when computing distance (only relevant if embedding is not None)
    :param embedding: compute distance in some embedded space or not; options are "pca", "lda", or None
    :return: distance

    """
    if module_feature_object.__class__.__name__ == "ModuleUsage":
        X = module_feature_object.label_counts
        obj_type = "module usage"
    elif module_feature_object.__class__.__name__ == "ModuleTransitions":
        X = module_feature_object.transition_counts
        obj_type = "module transitions"
    else:
        raise ValueError(
            f'module_feature_object class must be ModuleUsage or ModuleTransitions, not {module_feature_object.__class__}')

    distance_results = []

    if metric == "mahalanobis":
        lda = embed(module_feature_object, method="lda")
        cov = lda.covariance_

    if embedding == None:
        if hold_out:
            hold_out = False
            print("Ignoring hold_out=True because embedding is None")

    if not hold_out:
        if embedding != None:
            print(f"Warning - embedding (embedding={embedding}) without hold_out")
            if ((embedding == "lda") or (embedding == "LDA") or (embedding == "lineardiscriminantanalysis") or (
                    embedding == "LinearDiscriminantAnalysis")):
                emb = LDA(n_components=n_components, store_covariance=True)
                emb.fit(X, module_feature_object.group_labels)
                X = emb.transform(X)
            elif ((embedding == "pca") or (embedding == "PCA")):
                emb = PCA(n_components=n_components)
                emb.fit(X)
                X = emb.transform(X)

        if method == "pairwise":
            for i in range(X.shape[0]):
                for j in range(X.shape[0]):
                    if i < j:
                        grp_i = module_feature_object.group_labels[i]
                        grp_j = module_feature_object.group_labels[j]
                        obs_i = module_feature_object.observation_labels[i]
                        obs_j = module_feature_object.observation_labels[j]
                        if metric == "euclidean":
                            distance_ij = euclidean(X[i, :], X[j, :])
                        elif metric == "cityblock":
                            distance_ij = cityblock(X[i, :], X[j, :])
                        elif metric == "mahalanobis":
                            distance_ij = mahalanobis(X[i, :], X[j, :], cov)
                        else:
                            raise ValueError(f'metric must be euclidean, cityblock, or mahalanobis, not {metric}')
                        distance_results.append(pd.DataFrame(
                            {"grp_i": grp_i, "grp_j": grp_j, "obs_i": obs_i, "obs_j": obs_j,
                             "distance_ij": distance_ij},
                            index=[f'{grp_i}_{grp_j}']))
        elif method == "centroid":
            for group in sorted(np.unique(module_feature_object.group_labels)):
                centroid = np.mean(X[np.array(module_feature_object.group_labels) == group, :], axis=0)
                for i in range(X.shape[0]):
                    grp_i = module_feature_object.group_labels[i]
                    grp_j = group
                    obs_i = module_feature_object.observation_labels[i]
                    obs_j = f'centroid_{group}'
                    if metric == "euclidean":
                        distance_ij = euclidean(X[i, :], centroid)
                    elif metric == "cityblock":
                        distance_ij = cityblock(X[i, :], centroid)
                    elif metric == "mahalanobis":
                        distance_ij = mahalanobis(X[i, :], centroid, cov)
                    else:
                        raise ValueError(f'metric must be euclidean, cityblock, or mahalanobis, not {metric}')
                    distance_results.append(pd.DataFrame(
                        {"grp_i": grp_i, "grp_j": grp_j, "obs_i": obs_i, "obs_j": obs_j,
                         "distance_ij": distance_ij}, index=[f'{grp_i}_{grp_j}']))
        distance_results = pd.concat(distance_results, axis=0)
    else:
        for obs in range(len(module_feature_object.group_labels)):
            sub_X = np.delete(X, obs, axis=0)
            sub_y = module_feature_object.group_labels[:obs] + module_feature_object.group_labels[obs + 1:]
            X_i = X[obs, :]
            y_i = module_feature_object.group_labels[obs]
            if embedding != None:
                if ((embedding == "lda") or (embedding == "LDA") or (embedding == "lineardiscriminantanalysis") or (
                        embedding == "LinearDiscriminantAnalysis")):
                    emb = LDA(n_components=n_components, store_covariance=True)
                    emb.fit(sub_X, sub_y)
                    X = emb.transform(X)
                elif ((embedding == "pca") or (embedding == "PCA")):
                    emb = PCA(n_components=n_components)
                    emb.fit(sub_X)
                    X = emb.transform(X)

            if method == "pairwise":
                for i in range(X.shape[0]):
                    for j in range(X.shape[0]):
                        if i < j:
                            if ((j == obs) or (i == obs)):
                                grp_i = module_feature_object.group_labels[i]
                                grp_j = module_feature_object.group_labels[j]
                                obs_i = module_feature_object.observation_labels[i]
                                obs_j = module_feature_object.observation_labels[j]
                                if metric == "euclidean":
                                    distance_ij = euclidean(X[i, :], X[j, :])
                                elif metric == "cityblock":
                                    distance_ij = cityblock(X[i, :], X[j, :])
                                elif metric == "mahalanobis":
                                    distance_ij = mahalanobis(X[i, :], X[j, :], cov)
                                else:
                                    raise ValueError(
                                        f'metric must be euclidean, cityblock, or mahalanobis, not {metric}')
                                distance_results.append(pd.DataFrame(
                                    {"grp_i": grp_i, "grp_j": grp_j, "obs_i": obs_i, "obs_j": obs_j,
                                     "distance_ij": distance_ij}, index=[f'{grp_i}_{grp_j}']))
            elif method == "centroid":
                for group in sorted(np.unique(module_feature_object.group_labels)):
                    centroid = np.mean(X[np.array(module_feature_object.group_labels) == group, :], axis=0)
                    for i in range(X.shape[0]):
                        if i == obs:
                            grp_i = module_feature_object.group_labels[i]
                            grp_j = group
                            obs_i = module_feature_object.observation_labels[i]
                            obs_j = f'centroid_{group}'
                            if metric == "euclidean":
                                distance_ij = euclidean(X[i, :], centroid)
                            elif metric == "cityblock":
                                distance_ij = cityblock(X[i, :], centroid)
                            elif metric == "mahalanobis":
                                distance_ij = mahalanobis(X[i, :], centroid, cov)
                            else:
                                raise ValueError(
                                    f'metric must be euclidean, cityblock, or mahalanobis, not {metric}')
                            distance_results.append(pd.DataFrame(
                                {"grp_i": grp_i, "grp_j": grp_j, "obs_i": obs_i, "obs_j": obs_j,
                                 "distance_ij": distance_ij}, index=[f'{grp_i}_{grp_j}']))
        distance_results = pd.concat(distance_results, axis=0)
    return distance_results



    # def feat_select(usage_feats, method="f", n_feats=10, verbose=True):
#     """
#     A function for subselecting features that may be most relevant for classification
#
#     :param label_counts: label_counts array returned by analysis.get_usage_feats
#     :param group_labels: group_labels list returned by analysis.get_usage_feats
#     :param feat_names: feat_names list returned by analysis.get_usage_feats
#     :param method: Method for feature selection ("f", "pca")
#     :param n_feats: Number of features to select
#     :param verbose: Print output or not
#     :return:
#     """
#     n_groups=len(np.unique(usage_feats.group_labels))
#     picks=[]
#     if method=="pca":
#         if usage_feats.scaled == False:
#             usage_feats = usage_feats.scale()
#         pca = PCA(n_components=1)
#         pca.fit(usage_feats.label_counts)
#         picks = pca.components_.argsort()[0][0:n_feats]
#         text="Features selected by PCA were: "
#     elif method=="f":
#         group_data = []
#         for g in range(n_groups):
#             group_data.append(usage_feats.label_counts[np.array(usage_feats.group_labels) == g, :])
#         result = scipy.stats.f_oneway(*group_data)
#         stats = np.nan_to_num(result[0], nan=0)
#         ranked_stats = sorted(stats)
#         ranked_stats.reverse()
#         if n_feats>=len(ranked_stats):
#             picks=result[0]>-np.inf
#         else:
#             picks=result[0]>=ranked_stats[n_feats]
#         text="Features selected by f-stat were: "
#
#     n_picks=0
#     pick_names=[]
#     for p, pick in enumerate(picks):
#         if pick:
#             pick_names.append(usage_feats.feat_names[p])
#             n_picks=n_picks+1
#             if n_picks!=np.sum(picks):
#                 text+=str(usage_feats.feat_names[p])+", "
#             else:
#                 text+=str(usage_feats.feat_names[p])+"."
#
#
#     if verbose==True:
#         print(text)
#
#     return picks, pick_names
#
#

def regress(module_feature_object, dose_dict, method="LinearRegression", alpha = 1):
    if module_feature_object.__class__.__name__=="ModuleUsage":
        X=module_feature_object.label_counts
    elif module_feature_object.__class__.__name__=="ModuleTransitions":
        X=module_feature_object.transition_counts
    else:
        raise ValueError(f'module_feature_object class must be ModuleUsage or ModuleTransitions, not {module_feature_object.__class__}')
    reverse_grp_dict = {v: k for k, v in module_feature_object.group_dict.items()}
    dose_labels = [dose_dict[reverse_grp_dict[i]] for i in module_feature_object.group_labels]
    if method=="LinearRegression":
        reg = LinearRegression().fit(X, dose_labels)
    elif method=="Lasso":
        reg = Lasso(alpha=alpha).fit(X, dose_labels)
    elif method=="Ridge":
        reg = Ridge(alpha=alpha).fit(X, dose_labels)
    return reg, dose_labels

def loocv_regression(module_feature_object, dose_dict, method="LinearRegression", constrain_pos = True, alpha=1):
    loocv_preds = []
    if module_feature_object.__class__.__name__=="ModuleUsage":
        X=module_feature_object.label_counts
    elif module_feature_object.__class__.__name__=="ModuleTransitions":
        X=module_feature_object.transition_counts
    else:
        raise ValueError(f'module_feature_object class must be ModuleUsage or ModuleTransitions, not {module_feature_object.__class__}')
    reverse_grp_dict = {v: k for k, v in module_feature_object.group_dict.items()}
    y = [dose_dict[reverse_grp_dict[i]] for i in module_feature_object.group_labels]
    for obs in range(len(y)):
        sub_X = np.delete(X, obs, axis=0)
        sub_y = y[:obs] + y[obs + 1:]
        X_i = X[obs, :]
        y_i = y[obs]
        if method=="LinearRegression":
            reg = LinearRegression().fit(sub_X, sub_y)
        elif method=="Lasso":
            reg = Lasso(alpha=alpha).fit(sub_X, sub_y)
        elif method=="Ridge":
            reg = Ridge(alpha=alpha).fit(sub_X, sub_y)
        loocv_preds.append(reg.predict(X_i.reshape(1, -1))[0])
    loocv_preds = np.array(loocv_preds)
    if constrain_pos:
        loocv_preds[loocv_preds<0]=0
    y = np.array(y)
    sq_err = np.square(loocv_preds-y)
    return loocv_preds, sq_err

def get_usage_ssd(control_usage_feats, exp_usage_feats):
    """
    Get the time-resolved sum squared difference in module usage relative to a control distribution of usage.
    :param control_usage_feats: output from analysis.get_usage_feats for the control group ONLY
    :param exp_usage_feats:
    :return:
    """
    control_usage_feats_df = control_usage_feats.to_df()
    exp_usage_df = exp_usage_feats.to_df()
    control_usage_feats_df.drop("group", axis=1, inplace=True)
    exp_usage_df.drop("group", axis=1, inplace=True)
    control_usage = control_usage_feats_df.mean(axis=0)

    exp_usage_df_sqdiff = exp_usage_df.copy()

    for module in control_usage.index:
        mod_cols = [i for i in exp_usage_df.columns if module + '_' in i]
        exp_usage_df_sqdiff[mod_cols] = np.square(exp_usage_df_sqdiff[mod_cols] - control_usage[module])

    extracted = [col.split("_t")[1].split("-")[0] for col in exp_usage_df.columns]
    binstarts = pd.Series(extracted).unique()
    exp_usage_df_ssd = pd.DataFrame(index=exp_usage_df.index, columns=binstarts)

    for i in range(len(exp_usage_df_ssd.index)):
        for bin in binstarts:
            exp_usage_df_ssd.iloc[i][bin] = np.sum(exp_usage_df_sqdiff.iloc[i].filter(like="_t" + bin + "-"))

    return exp_usage_df_ssd

# class LdaResult:
#     def __init__(self, lda, lda_embeddings, label_counts, group_labels, feat_picks, feat_names, group_dict, nbins, binsize,
#                  loocv_accuracy,loocv_confmat):
#         self.lda = lda
#         self.lda_embeddings = lda_embeddings
#         self.label_counts = label_counts
#         self.group_labels = group_labels
#         self.feat_picks = feat_picks
#         self.feat_names = feat_names
#         self.group_dict = group_dict
#         self.nbins = nbins
#         self.binsize = binsize
#         self.loocv_accuracy = loocv_accuracy
#         self.loocv_confmat = loocv_confmat
#
#     def get_discriminant_weights(self):
#         n_feats = np.sum(self.feat_picks)
#         n_components = len(self.lda.explained_variance_ratio_)
#         LD_weightings = np.zeros([n_feats,n_components])
#         for feat in range(n_feats):
#             test=np.zeros(n_feats)
#             test[feat]=1
#             LD_weightings[feat,:] = self.lda.transform([test])[0]
#         LD_names=[f"LD{i+1}" for i in range(n_components)]
#         LD_weightings = pd.DataFrame(LD_weightings,index=self.feat_names,columns=LD_names)
#
#         # Test function, no longer needed
#         # cmap = plt.get_cmap("viridis_r")
#         # colors = [cmap([i]) for i in np.linspace(0,1,len(selected_subgroups))]
#         #
#         # data = lda_result.label_counts[:,lda_result.feat_picks]
#         # for d in range(data.shape[0]):
#         #     x=np.sum(LD_weightings[:,0]*data[d,:])
#         #     y=np.sum(LD_weightings[:,1]*data[d,:])
#         #     plt.scatter(x,y,color = colors[lda_result.group_labels[d]])
#
#         return(LD_weightings)


    # def get_mahalanobis_distance(self,type="point_to_point"):
    #     """
    #     Calculates Mahalanobis distance between groups
    #
    #     :param type: Whether to do pairwise point to point ("point_to_point") distances or point to centroid ("point_to_centroid") distances
    #     :return: a dictionary of pairwise distances
    #     """
    #     dists = {}
    #     for k1_ind, k1 in enumerate(list(self.group_dict.keys())):
    #         for k2_ind, k2 in enumerate(list(self.group_dict.keys())):
    #             if k1_ind >= k2_ind and type=="point_to_point":
    #                 continue
    #             else:
    #                 mas = [i == self.group_dict[k1] for i in self.group_labels]
    #                 g1 = self.label_counts[:, self.feat_picks][mas, :]
    #                 m1 = np.mean(g1, axis=0)
    #                 mas = [i == self.group_dict[k2] for i in self.group_labels]
    #                 g2 = self.label_counts[:, self.feat_picks][mas, :]
    #
    #                 if type=="point_to_point":
    #                     dists_k1k2 = []
    #                     if k1 != k2:
    #                         for i in range(g1.shape[0]):
    #                             for j in range(g2.shape[0]):
    #                                 dists_k1k2.append(scipy.spatial.distance.mahalanobis(g1[i], g2[j], self.lda.covariance_))
    #                     else:
    #                         for i in range(g1.shape[0]):
    #                             for j in range(g2.shape[0]):
    #                                 if j > i:
    #                                     dists_k1k2.append(scipy.spatial.distance.mahalanobis(g1[i], g2[j], self.lda.covariance_))
    #                     dists[k1 + "____vs____" + k2] = dists_k1k2
    #                 elif type=="point_to_centroid":
    #                     dists_k1k2 = []
    #                     if k1 != k2:
    #                         for j in range(g2.shape[0]):
    #                             dists_k1k2.append(scipy.spatial.distance.mahalanobis(m1, g2[j], self.lda.covariance_))
    #                     else:
    #                         for j in range(g2.shape[0]):
    #                                 dists_k1k2.append(scipy.spatial.distance.mahalanobis(m1, g2[j], self.lda.covariance_))
    #                     dists[k1 + "-CENTROID____vs____" + k2] = dists_k1k2
    #                 else:
    #                     raise ValueError("Invalid comparison type! Comparison must be 'point_to_point' or 'point_to_centroid'.")
    #
    #     return dists


# def lda_labels_timebins(config,
#                         labels_df,
#                         binsize,
#                         selected_subgroups="all",
#                         ncomponents=2,
#                         feature_selection=None,
#                         loocv=False, scale=True):
#     """
#     Function to compute LDA for data in timebins
#
#     :param config: config object
#     :param labels_df: labels_df object
#     :param binsize: bin width in seconds
#     :param selected_subgroups:
#     :param ncomponents: number of linear discriminants
#     :param feature_selection: None for no feature selection or tuple of method "pca","f" and number of features
#     :param loocv: do leave-one-out cross-validation
#     :return:
#     """
#     if selected_subgroups=="all":
#         selected_subgroups=list(config["subgroups"].keys())
#     n_groups = len(selected_subgroups)
#     fps = int(config["fps"])
#     group_dict = {selected_subgroups[i]: i for i in range(len(selected_subgroups))}
#     nbins = int(labels_df.shape[0] / (binsize * fps))
#     module_usage = get_module_usage(config,labels_df,binsize,selected_subgroups=selected_subgroups)
#     if scale==True:
#         usage_feats = module_usage.scale()
#     label_counts_full = module_usage.label_counts.copy()
#     group_labels_full = module_usage.group_labels.copy()
#     if feature_selection is not None:
#         feat_picks, feat_names = feat_select(module_usage, method=feature_selection[0], n_feats=feature_selection[1])
#         label_counts = module_usage.label_counts[:, feat_picks]
#     else:
#         feat_names = module_usage.feat_names
#         feat_picks = [True for feat in feat_names]
#         label_counts = module_usage.label_counts
#     lda = LDA(n_components=ncomponents,store_covariance=True)
#     lda_embeddings = lda.fit_transform(label_counts, module_usage.group_labels)
#
#     if loocv==True:
#         predictions=[]
#         true_class=[]
#         for sample_i in range(label_counts_full.shape[0]):
#             label_counts_sub=np.delete(label_counts_full,sample_i,axis=0)
#             label_counts_i=label_counts_full[sample_i,:]
#             group_labels_sub=group_labels_full.copy()
#             label_i=group_labels_sub.pop(sample_i)
#             if feature_selection is not None:
#                 picks, _ = feat_select(module_usage, method=feature_selection[0], n_feats=feature_selection[1], verbose=False)
#                 label_counts_sub = label_counts_sub[:, picks]
#                 label_counts_i = label_counts_i[picks]
#             lda_sub = LDA(n_components=ncomponents)
#             lda_sub.fit(label_counts_sub, group_labels_sub)
#
#             pred=lda_sub.predict(label_counts_i.reshape(1, -1))[0]
#             predictions.append(pred)
#             true_class.append(label_i)
#         predictions=np.array(predictions)
#         true_class=np.array(true_class)
#         classes=np.unique(true_class)
#         loocv_confmat=np.zeros([len(classes),len(classes)])
#         for pred_i in range(len(predictions)):
#             loocv_confmat[predictions[pred_i],true_class[pred_i]]+=1
#         loocv_accuracy=np.mean(predictions==true_class)
#     else:
#         loocv_accuracy="Cross-validation not completed"
#         loocv_confmat="Cross-validation not completed"
#
#     return LdaResult(lda, lda_embeddings, module_usage.label_counts, module_usage.group_labels, feat_picks, feat_names,
#                      group_dict, nbins, binsize, loocv_accuracy, loocv_confmat)

# def lda_labels_timebins(config,
#                         labels_df,
#                         binsize,
#                         selected_subgroups="all",
#                         ncomponents=2,
#                         feature_selection=None,
#                         loocv=False, scale=True):
#     """
#     Function to compute LDA for data in timebins
#
#     :param config: config object
#     :param labels_df: labels_df object
#     :param binsize: bin width in seconds
#     :param selected_subgroups:
#     :param ncomponents: number of linear discriminants
#     :param feature_selection: None for no feature selection or tuple of method "pca","f" and number of features
#     :param loocv: do leave-one-out cross-validation
#     :return:
#     """
#     if selected_subgroups=="all":
#         selected_subgroups=list(config["subgroups"].keys())
#     n_groups = len(selected_subgroups)
#     fps = int(config["fps"])
#     group_dict = {selected_subgroups[i]: i for i in range(len(selected_subgroups))}
#     nbins = int(labels_df.shape[0] / (binsize * fps))
#     module_usage = get_module_usage(config,labels_df,binsize,selected_subgroups=selected_subgroups)
#     if scale==True:
#         usage_feats = module_usage.scale()
#     label_counts_full = module_usage.label_counts.copy()
#     group_labels_full = module_usage.group_labels.copy()
#     if feature_selection is not None:
#         feat_picks, feat_names = feat_select(module_usage, method=feature_selection[0], n_feats=feature_selection[1])
#         label_counts = module_usage.label_counts[:, feat_picks]
#     else:
#         feat_names = module_usage.feat_names
#         feat_picks = [True for feat in feat_names]
#         label_counts = module_usage.label_counts
#     lda = LDA(n_components=ncomponents,store_covariance=True)
#     lda_embeddings = lda.fit_transform(label_counts, module_usage.group_labels)
#
#     if loocv==True:
#         predictions=[]
#         true_class=[]
#         for sample_i in range(label_counts_full.shape[0]):
#             label_counts_sub=np.delete(label_counts_full,sample_i,axis=0)
#             label_counts_i=label_counts_full[sample_i,:]
#             group_labels_sub=group_labels_full.copy()
#             label_i=group_labels_sub.pop(sample_i)
#             if feature_selection is not None:
#                 picks, _ = feat_select(module_usage, method=feature_selection[0], n_feats=feature_selection[1], verbose=False)
#                 label_counts_sub = label_counts_sub[:, picks]
#                 label_counts_i = label_counts_i[picks]
#             lda_sub = LDA(n_components=ncomponents)
#             lda_sub.fit(label_counts_sub, group_labels_sub)
#
#             pred=lda_sub.predict(label_counts_i.reshape(1, -1))[0]
#             predictions.append(pred)
#             true_class.append(label_i)
#         predictions=np.array(predictions)
#         true_class=np.array(true_class)
#         classes=np.unique(true_class)
#         loocv_confmat=np.zeros([len(classes),len(classes)])
#         for pred_i in range(len(predictions)):
#             loocv_confmat[predictions[pred_i],true_class[pred_i]]+=1
#         loocv_accuracy=np.mean(predictions==true_class)
#     else:
#         loocv_accuracy="Cross-validation not completed"
#         loocv_confmat="Cross-validation not completed"
#
#     return LdaResult(lda, lda_embeddings, module_usage.label_counts, module_usage.group_labels, feat_picks, feat_names,
#                      group_dict, nbins, binsize, loocv_accuracy, loocv_confmat)
#
# def lda_loco_labels_timebins(config, dist_df, ncomponents=2):
#     """
#     Function to compute LDA for loco/kepoint position data in timebins
#
#     :param config:
#     :param labels_df:
#     :param binsize:
#     :param selected_subgroups:
#     :param ncomponents:
#     :return:
#     """
#     selected_subgroups = pd.Series([i[0] for i in dist_df.columns]).unique()
#     n_groups = len(selected_subgroups)
#     fps = int(config["fps"])
#     group_dict = {selected_subgroups[i]: i for i in range(len(selected_subgroups))}
#
#     dist_counts = []
#     for g in range(n_groups):
#         for i in range(len(dist_df[selected_subgroups[g]].columns)):
#             sub_df=dist_df[selected_subgroups[g]]
#             dist_counts.append(sub_df[sub_df.columns[i]])
#     dist_counts = np.array(dist_counts)
#
#     group_labels = []
#     for g in range(n_groups):
#         group_labels.extend([g] * len(dist_df[selected_subgroups[g]].columns))
#
#     lda = LDA(n_components=ncomponents)
#     lda_embeddings = lda.fit_transform(dist_counts, group_labels)
#     nbins=dist_df.shape[0]
#     return lda, lda_embeddings, dist_counts, group_labels, group_dict, nbins
#
#
# def lr_labels_timebins(config, labels_df, binsize, selected_subgroups="all"):
#     """
#     Function to compute LDA for data in timebins
#
#     :param config:
#     :param labels_df:
#     :param binsize:
#     :param selected_subgroups:
#     :param ncomponents:
#     :return:
#     """
#     # TODO:: build out logistic regression classification function
#     # if groupnames == None:
#     #     groupnames = list(np.unique([item[0] for item in labels_df.columns]))
#     if selected_subgroups=="all":
#         selected_subgroups=list(config["subgroups"].keys())
#     n_groups = len(selected_subgroups)
#     # n_groups = len(list(np.unique([item[0] for item in labels_df.columns])))
#     fps = int(config["fps"])
#     # start_frame = start * fps
#     # stop_frame = stop * fps
#     # labels_df = labels_df[start_frame:stop_frame]
#     # total_frames = stop_frame - start_frame
#     group_dict = {selected_subgroups[i]: i for i in range(len(selected_subgroups))}
#     labels_flat = np.array(labels_df)
#     labels_flat = [item for sublist in labels_flat for item in sublist]
#     clusts = np.unique(labels_flat)
#     n_clust = len(clusts)
#
#     label_counts = []
#     nbins = int(labels_df.shape[0] / (binsize * fps))
#     for g in range(n_groups):
#         for i in range(len(labels_df[selected_subgroups[g]].columns)):
#             label_counts_i = np.zeros(n_clust * nbins)
#             for b in range(nbins):
#                 binstart = int(b * (binsize * fps))
#                 binstop = int((b + 1) * (binsize * fps))
#                 labels_df_sub = labels_df[binstart:binstop]
#                 for c in range(n_clust):
#                     label_counts_i[c + n_clust * b] = np.count_nonzero(
#                         labels_df_sub[selected_subgroups[g]][[labels_df[selected_subgroups[g]].columns[i]]] == c) / (5 * 60 * fps)
#             label_counts.append(label_counts_i)
#     label_counts = np.array(label_counts)
#
#     group_labels = []
#     for g in range(n_groups):
#         group_labels.extend([g] * len(labels_df[selected_subgroups[g]].columns))
#
#     lr = LR().fit(label_counts, group_labels)
#     return lr, group_labels, label_counts, group_dict, nbins
#
# def nlp_classification(config, labels_df):
#     print("Coming soon!")
#     # TODO:: add NLP classification function