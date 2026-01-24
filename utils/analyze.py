import numpy as np
import pandas as pd
import os
import copy
from sklearn.tree import plot_tree
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression as LR
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from scipy import stats
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from itertools import combinations
import pickle
import h5py

import scipy
from scipy.spatial import distance

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

def pickle_dump(object, save_path):
    if ((save_path.endswith(".pkl")==False) and (save_path.endswith(".pickle")==False)):
        save_path+=".pickle"
        print(f"Save path changed to {save_path}")
    with open(save_path, 'wb') as file:
        pickle.dump(object, file)

def pickle_load(object_path):
    with open(object_path, 'rb') as file:
        obj = pickle.load(file)
    return obj

class KeypointFeature:
    def __init__(self, keypoint_feature, group_labels, observation_labels, feat_names, group_dict, scaler, feature_type):
        self.keypoint_feature = keypoint_feature
        self.group_labels = group_labels
        self.observation_labels = observation_labels
        self.feat_names = feat_names
        self.group_dict = group_dict
        self.scaler = scaler
        mean_check = np.allclose(np.mean(keypoint_feature, axis=0), 0, atol=0.1)
        std_check = np.allclose(np.std(keypoint_feature, axis=0), 1, atol=0.1)
        self.scaled = mean_check and std_check
        self.feature_type = feature_type

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
        df = pd.DataFrame(self.keypoint_feature, columns=self.feat_names, index=self.observation_labels)
        df["group"] = colnames
        return df

    def scale(self):
        scaler = StandardScaler()
        keypoint_feature_scaled = scaler.fit_transform(self.keypoint_feature)
        return KeypointFeature(keypoint_feature_scaled, self.group_labels, self.observation_labels, self.feat_names,
                           self.group_dict, scaler, self.feature_type)

    def save(self, save_path):
        if ((save_path.endswith(".pkl")==False) and (save_path.endswith(".pickle")==False)):
            save_path+=".pickle"
            print(f"Save path changed to {save_path}")
        with open(save_path, 'wb') as file:
            pickle.dump(self, file)


def get_module_labels(config, start, stop, subgroups = None):
    """
    Generates a Pandas dataframe containing the labels for every frame in the specified time range for the video paths in defined groups.

    :param config: the config
    :param start: time in seconds to start dataframe from.
    :param stop: time in seconds to stop dataframe at.
    :param fps: frames per second of recording.
    :param subgroups: subgroups to include; by default, None will result in an object without data subgrouped; could alternatively be a list of subgroup names from config or "all" (to include all subgroups present in config)
    :return: labels dataframe
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
            model_path = "/VAME/"+config["vame_model_name"]+"/"
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
            model_path = "/VAME/"+config["vame_model_name"]+"/"
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

def get_keypoint_travel(PE_config,
                        keypoint,
                        start,
                        end,
                        binsize=None,
                        thresh=70,
                        selected_subgroups=None,
                        return_as_df=True):
    """
    Measure keypodint distance travelled across group of subjects

    :param PE_config: pose estimation config - used for FPS only
    :param keypoint: name of keypoint
    :param start: start time in seconds
    :param end: end time in seconds
    :param binsize: if binned output is desired, size of timebins (default is None for no binning)
    :param thresh: threshold for when distance 'jump' is too large and should be excluded; default 70
    :param selected_subgroups: subgroups to analyze as defined in config; default "all"
    :param return_as_df: return as a dataframe or return as KeypointTravel class (for embedding, classification)
    :return:
    """
    if selected_subgroups == None:
        if selected_subgroups == "all":
            selected_subgroups = PE_config["subgroups"].keys()

        group_dict = {"no_assigned_subgroup": 0}

        group_labels = []
        travel_data = []
        observation_labels = []
        for s, sess in enumerate(PE_config["project_files"]):
            observation_labels.append(sess)
            group_labels.append(0)
            filepath = PE_config["data_directory"] + "/" + sess
            if PE_config["data_source"] == "DeepLabCut":
                data = pd.read_csv(filepath, header=[1, 2])
            elif PE_config["data_source"] == "OpenFace":
                data = read_openface_csv(PE_config, filepath)
            elif PE_config["data_source"] == "SLEAP":
                final_filepath, track_name = tuple(filepath.split(":::"))
                data = read_sleap_csv(final_filepath, track_name)
            if binsize == None:
                binsize = end - start
                nbins = 1
            else:
                nbins = int((end - start) / binsize)
            dist_i = np.zeros(nbins)
            fps = int(PE_config["fps"])
            for b in range(nbins):
                x = np.array(data[keypoint]['x'])[(start + b * binsize) * fps:(start + (b + 1) * binsize) * fps]
                y = np.array(data[keypoint]['y'])[(start + b * binsize) * fps:(start + (b + 1) * binsize) * fps]
                for i in range(len(x) - 1):
                    if np.absolute(x[i + 1] - x[i]) > thresh:
                        x[i + 1] = x[i]
                    if np.absolute(y[i + 1] - y[i]) > thresh:
                        y[i + 1] = y[i]
                    dist_i[b] = dist_i[b] + np.sqrt((x[i + 1] - x[i]) ** 2 + (y[i + 1] - y[i]) ** 2)
            travel_data.append(dist_i)
        travel_data = np.array(travel_data)
        if binsize is not None:
            feat_names = np.arange(start / 60, end / 60, binsize / 60)
        else:
            feat_names = "travel"

    else:
        if selected_subgroups=="all":
            selected_subgroups=PE_config["subgroups"].keys()

        group_dict = {selected_subgroups[i]: i for i in range(len(selected_subgroups))}

        group_labels=[]
        travel_data=[]
        observation_labels=[]
        for g, group in enumerate(selected_subgroups):
            for s, sess in enumerate(PE_config["subgroups"][group]):
                observation_labels.append(sess)
                group_labels.append(group_dict[group])
                filepath = PE_config["data_directory"]+"/"+sess
                if PE_config["data_source"]=="DeepLabCut":
                    data = pd.read_csv(filepath, header=[1, 2])
                elif PE_config["data_source"]=="OpenFace":
                    data = read_openface_csv(PE_config, filepath)
                elif PE_config["data_source"]=="SLEAP":
                    final_filepath, track_name = tuple(filepath.split(":::"))
                    data = read_sleap_csv(final_filepath, track_name)
                if binsize == None:
                    binsize = end - start
                    nbins = 1
                else:
                    nbins = int((end - start) / binsize)
                dist_i = np.zeros(nbins)
                fps = int(PE_config["fps"])
                for b in range(nbins):
                    x = np.array(data[keypoint]['x'])[(start + b * binsize) * fps:(start + (b + 1) * binsize) * fps]
                    y = np.array(data[keypoint]['y'])[(start + b * binsize) * fps:(start + (b + 1) * binsize) * fps]
                    for i in range(len(x) - 1):
                        if np.absolute(x[i + 1] - x[i]) > thresh:
                            x[i + 1] = x[i]
                        if np.absolute(y[i + 1] - y[i]) > thresh:
                            y[i + 1] = y[i]
                        dist_i[b] = dist_i[b] + np.sqrt((x[i + 1] - x[i]) ** 2 + (y[i + 1] - y[i]) ** 2)
                travel_data.append(dist_i)
        travel_data = np.array(travel_data)
        if binsize is not None:
            feat_names=np.arange(start/60,end/60,binsize/60)
        else:
            feat_names="travel"
    if return_as_df:
        kf = KeypointFeature(travel_data, group_labels, observation_labels, feat_names, group_dict, None, "travel")
        kf_df = kf.to_df()
        return kf_df
    else:
        return KeypointFeature(travel_data, group_labels, observation_labels, feat_names, group_dict, None, "travel")


def ego_center(config, data, keypoint_ego1, keypoint_ego2):
    """
    Ego centering a single pose estimation dataframe

    :param config: config
    :param data: pose estimation pandas dataframe in DLC format
    :param keypoint_ego1: first keypoint to use for establishing egocentric alignment axis
    :param keypoint_ego2: second keypoint to use for establishing egocentric alignment axis
    :return: ego_centered pandas dataframe

    """
    x_coords = data.xs("x", level=1, axis=1)
    y_coords = data.xs("y", level=1, axis=1)
    ego1 = np.stack([x_coords[keypoint_ego1], y_coords[keypoint_ego1]], axis=-1)
    ego2 = np.stack([x_coords[keypoint_ego2], y_coords[keypoint_ego2]], axis=-1)
    origin = (ego1 + ego2) / 2
    delta = ego2 - ego1
    theta = np.arctan2(delta[:, 0], delta[:, 1])
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    new_data = {}

    for keypoint in config["keypoints"]:
        point = np.stack([x_coords[keypoint], y_coords[keypoint]], axis=-1)
        new_point = point - origin
        rot_x = new_point[:, 0] * cos_theta - new_point[:, 1] * sin_theta
        rot_y = new_point[:, 0] * sin_theta + new_point[:, 1] * cos_theta
        new_data[(keypoint, "x")] = rot_x
        new_data[(keypoint, "y")] = rot_y

    return pd.DataFrame(new_data, index=data.index)

def read_openface_csv(of_config, filepath):
    data = pd.read_csv(filepath)
    data.columns = data.columns.str.replace(' ', '')

    data_dlc = {}
    for keypoint in of_config["keypoints"]:
        keypoints_x = ["x" + keypoint.split("kp")[1], "X" + keypoint.split("kp")[1]]
        keypoints_x = [i for i in keypoints_x if i in data.columns]
        data_dlc[(keypoint, "x")] = data[keypoints_x].to_numpy().T[0]
        keypoints_y = ["y" + keypoint.split("kp")[1], "Y" + keypoint.split("kp")[1]]
        keypoints_y = [i for i in keypoints_y if i in data.columns]
        data_dlc[(keypoint, "y")] = data[keypoints_y].to_numpy().T[0]

    return pd.DataFrame(data_dlc)

def read_sleap_csv(filepath, track_name):
    with h5py.File(filepath, "r") as f:
        node_names = f['node_names'][:]
        node_names = [node.decode('utf-8') for node in node_names]
        track_names = f['track_names'][:]
        track_names = [tr.decode('utf-8') for tr in track_names]
        track_occupancy = f['track_occupancy'][:]
        tracks = f['tracks'][:]

    track_idx = np.argmax([i == track_name for i in track_names])

    df = {}
    for n, node in enumerate(node_names):
        df[(node, "x")] = tracks[track_idx][0][n]
        df[(node, "y")] = tracks[track_idx][1][n]
    df = pd.DataFrame(df)

    return pd.DataFrame(df)
def get_keypoint_kinematics(PE_config,
                            keypoint_ego1,
                            keypoint_ego2,
                            start,
                            end,
                            binsize=None,
                            metric="angle",
                            thresh=70,
                            selected_subgroups=None,
                            return_as_df=True, verbose=False):
    """
    Measure keypoint kinematics (egocentrically aligned angle, distance, and travel of keypoints) across group of subjects

    :param PE_config: pose estimation config - used for FPS only
    :param keypoint1: first keypoint to use for establishing egocentric alignment axis
    :param keypoint2: second keypoint to use for establishing egocentric alignment axis
    :param start: start time in seconds
    :param end: end time in seconds
    :param binsize: if binned output is desired, size of timebins (default is None for no binning)
    :param metric: metric to assess for all non-keypoint1/2 keypoints; options are "angle_m" (for mean angle), "angle_sd" (for std deviation of angle), "distance_m" (for mean distance of keypoints to ego-center), "distance_sd" (for std deviation of distance of keypoints to ego-center), or "travel" (for movement of keypoints relative to ego-center)
    :param thresh: threshold for when distance 'jump' is too large and should be excluded; default 70
    :param selected_subgroups: subgroups to analyze as defined in config; default "all"
    :param return_as_df: return as a dataframe or return as KeypointTravel class (for embedding, classification)
    :return:

    """

    if metric=="travel":
        feature_type = "Keypoint travel relative to ego-center"
    elif metric=="angle_m":
        feature_type = "Keypoint mean angle relative to ego-center"
    elif metric=="angle_sd":
        feature_type = "Keypoint std deviation angle relative to ego-center"
    elif metric=="distance_m":
        feature_type =  "Keypoint mean distance to ego-center"
    elif metric=="distance_sd":
        feature_type =  "Keypoint std deviation distance to ego-center"
    else:
        return ValueError(f"Invalid metric ({metric}) - must be one of: 'angle_m' (for mean angle), 'angle_sd' (for std deviation of angle), 'distance_m' (for mean distance of keypoints to ego-center), 'distance_sd' (for std deviation of distance of keypoints to ego-center), or 'travel' (for movement of keypoints relative to ego-center)")

    if selected_subgroups==None:

        group_dict = {"no_assigned_subgroup": 0}

        non_ego_keypoints = [i for i in PE_config["keypoints"] if ((i != keypoint_ego1) and (i != keypoint_ego2))]

        group_labels = []
        kinematic_data = []
        observation_labels = []

        n_observations = len(PE_config["project_files"])

        count = 0
        for s, sess in enumerate(PE_config["project_files"]):
            count += 1
            observation_labels.append(sess)
            if verbose:
                print(f"Working on {metric} kinematics for observation {count} of {n_observations}")
            group_labels.append(0)
            filepath = PE_config["data_directory"] + "/" + sess
            if PE_config["data_source"] == "DeepLabCut":
                data = pd.read_csv(filepath, header=[1, 2])
                if verbose:
                    print("Ego-centering")
                data_new = ego_center(PE_config, data, keypoint_ego1, keypoint_ego2)
                data = data_new
            elif PE_config["data_source"] == "OpenFace":
                data = read_openface_csv(PE_config, filepath)
                if verbose:
                    print("Ego-centering")
                data_new = ego_center(PE_config, data, keypoint_ego1, keypoint_ego2)
                data = data_new
            elif PE_config["data_source"] == "SLEAP":
                final_filepath, track_name = tuple(filepath.split(":::"))
                data = read_sleap_csv(final_filepath, track_name)
                if verbose:
                    print("Ego-centering")
                data_new = ego_center(PE_config, data, keypoint_ego1, keypoint_ego2)
                data = data_new
            if binsize == None:
                binsize = end - start
                nbins = 1
            else:
                nbins = int((end - start) / binsize)
            kin_i = np.zeros(nbins * len(non_ego_keypoints))
            fps = int(PE_config["fps"])

            if verbose:
                print(f"Computing {metric}")
            for kp, keypoint in enumerate(non_ego_keypoints):
                for b in range(nbins):
                    x = np.array(data[keypoint]['x'])[(start + b * binsize) * fps:(start + (b + 1) * binsize) * fps]
                    y = np.array(data[keypoint]['y'])[(start + b * binsize) * fps:(start + (b + 1) * binsize) * fps]
                    for i in range(len(x) - 1):
                        if np.absolute(x[i + 1] - x[i]) > thresh:
                            x[i + 1] = x[i]
                        if np.absolute(y[i + 1] - y[i]) > thresh:
                            y[i + 1] = y[i]
                        if metric == "travel":
                            kin_i[(kp * nbins) + b] = kin_i[(kp * nbins) + b] + np.sqrt(
                                (x[i + 1] - x[i]) ** 2 + (y[i + 1] - y[i]) ** 2)
                    if metric == "angle_m":
                        kin_i[(kp * nbins) + b] = np.mean(np.arctan2(y, x))
                    if metric == "angle_sd":
                        kin_i[(kp * nbins) + b] = np.std(np.arctan2(y, x))
                    elif metric == "distance_m":
                        kin_i[(kp * nbins) + b] = np.mean(np.sqrt((x ** 2) + (y ** 2)))
                    elif metric == "distance_sd":
                        kin_i[(kp * nbins) + b] = np.std(np.sqrt((x ** 2) + (y ** 2)))
            kinematic_data.append(kin_i)
        kinematic_data = np.array(kinematic_data)
        feat_names = []
        for kp in non_ego_keypoints:
            if binsize is not None:
                feat_name_kp = np.arange(start / 60, end / 60, binsize / 60)
                feat_names += [metric + "_" + kp + "_" + str(i) for i in feat_name_kp]
            else:
                feat_names.append(metric + "_" + kp)
    else:
        if selected_subgroups == "all":
            selected_subgroups = PE_config["subgroups"].keys()

        group_dict = {selected_subgroups[i]: i for i in range(len(selected_subgroups))}

        non_ego_keypoints = [i for i in PE_config["keypoints"] if ((i != keypoint_ego1) and (i != keypoint_ego2))]

        group_labels = []
        kinematic_data = []
        observation_labels = []

        n_observations = np.sum([len(PE_config["subgroups"][group]) for group in selected_subgroups])

        count = 0
        for g, group in enumerate(selected_subgroups):
            for s, sess in enumerate(PE_config["subgroups"][group]):
                count += 1
                observation_labels.append(sess)
                if verbose:
                    print(f"Working on {metric} kinematics for observation {count} of {n_observations}")
                group_labels.append(group_dict[group])
                filepath = PE_config["data_directory"] + "/" + sess
                if PE_config["data_source"] == "DeepLabCut":
                    data = pd.read_csv(filepath, header=[1, 2])
                    if verbose:
                        print("Ego-centering")
                    data_new = ego_center(PE_config, data, keypoint_ego1, keypoint_ego2)
                    data = data_new
                elif PE_config["data_source"] == "OpenFace":
                    data = read_openface_csv(PE_config,filepath)
                    if verbose:
                        print("Ego-centering")
                    data_new = ego_center(PE_config, data, keypoint_ego1, keypoint_ego2)
                    data = data_new
                elif PE_config["data_source"]=="SLEAP":
                    final_filepath, track_name = tuple(filepath.split(":::"))
                    data = read_sleap_csv(final_filepath, track_name)
                    if verbose:
                        print("Ego-centering")
                    data_new = ego_center(PE_config, data, keypoint_ego1, keypoint_ego2)
                    data = data_new
                if binsize == None:
                    binsize = end - start
                    nbins = 1
                else:
                    nbins = int((end - start) / binsize)
                kin_i = np.zeros(nbins * len(non_ego_keypoints))
                fps = int(PE_config["fps"])

                if verbose:
                    print(f"Computing {metric}")
                for kp, keypoint in enumerate(non_ego_keypoints):
                    for b in range(nbins):
                        x = np.array(data[keypoint]['x'])[(start + b * binsize) * fps:(start + (b + 1) * binsize) * fps]
                        y = np.array(data[keypoint]['y'])[(start + b * binsize) * fps:(start + (b + 1) * binsize) * fps]
                        for i in range(len(x) - 1):
                            if np.absolute(x[i + 1] - x[i]) > thresh:
                                x[i + 1] = x[i]
                            if np.absolute(y[i + 1] - y[i]) > thresh:
                                y[i + 1] = y[i]
                            if metric == "travel":
                                kin_i[(kp * nbins) + b] = kin_i[(kp * nbins) + b] + np.sqrt(
                                    (x[i + 1] - x[i]) ** 2 + (y[i + 1] - y[i]) ** 2)
                        if metric == "angle_m":
                            kin_i[(kp * nbins) + b] = np.mean(np.arctan2(y, x))
                        if metric == "angle_sd":
                            kin_i[(kp * nbins) + b] = np.std(np.arctan2(y, x))
                        elif metric == "distance_m":
                            kin_i[(kp * nbins) + b] = np.mean(np.sqrt((x ** 2) + (y ** 2)))
                        elif metric == "distance_sd":
                            kin_i[(kp * nbins) + b] = np.std(np.sqrt((x ** 2) + (y ** 2)))
                kinematic_data.append(kin_i)
        kinematic_data = np.array(kinematic_data)
        feat_names = []
        for kp in non_ego_keypoints:
            if binsize is not None:
                feat_name_kp = np.arange(start / 60, end / 60, binsize / 60)
                feat_names += [metric + "_" + kp + "_" + str(i) for i in feat_name_kp]
            else:
                feat_names.append(metric + "_" + kp)
    if return_as_df:
        kf = KeypointFeature(kinematic_data, group_labels, observation_labels, feat_names, group_dict, None, feature_type)
        kf_df = kf.to_df()
        return kf_df
    else:
        return KeypointFeature(kinematic_data, group_labels, observation_labels, feat_names, group_dict, None, feature_type)


class ActionUnits:
    def __init__(self, action_units, group_labels, observation_labels, feat_names, group_dict, scaler):
        self.action_units = action_units
        self.group_labels = group_labels
        self.observation_labels = observation_labels
        self.feat_names = feat_names
        self.group_dict = group_dict
        self.scaler = scaler
        mean_check = np.allclose(np.mean(action_units, axis=0), 0, atol=0.1)
        std_check = np.allclose(np.std(action_units, axis=0), 1, atol=0.1)
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
        df = pd.DataFrame(self.action_units, columns=self.feat_names, index=self.observation_labels)
        df["group"] = colnames
        return df

    def scale(self):
        scaler = StandardScaler()
        action_units_scaled = scaler.fit_transform(self.action_units)
        return ActionUnits(action_units_scaled, self.group_labels, self.observation_labels, self.feat_names,
                           self.group_dict, scaler)

    def save(self, save_path):
        if ((save_path.endswith(".pkl")==False) and (save_path.endswith(".pickle")==False)):
            save_path+=".pickle"
            print(f"Save path changed to {save_path}")
        with open(save_path, 'wb') as file:
            pickle.dump(self, file)


def get_action_units(config, start, end, binsize=None, selected_subgroups=None, aus_to_include="all"):
    """
    Function for getting mean action units
    :param config: project config
    :param start: start time (in s)
    :param end: end time (in s)
    :param binsize: optional - temporal bin size (default None for no binning)
    :param selected_subgroups: otpional - selected subgroups
    :param aus_to_include: list of integers corresponding to action units to include, or "all" for all; default "all"
    :return: ActionUnits

    """
    all_au_data = {}
    group_labels = []
    observation_labels = []

    if binsize == None:
        binsize = end - start
        nbins = 1
    else:
        nbins = int((end - start) / binsize)

    if selected_subgroups == None:
        warning_printed=False
        group_dict = {"no_assigned_subgroup": 0}
        for subj in config["project_files"]:
            df = pd.read_csv(config["data_directory"] + "/" + subj)
            df.columns = df.columns.str.replace(' ', '')
            au_cols = [i for i in df.columns if ((i.startswith("AU")) and (i.endswith("r")))]
            if aus_to_include!="all":
                au_cols_subset = []
                full_au_cols = au_cols.copy()
                bad_aus = []
                for au_idx in aus_to_include:
                    au_to_add = [au_i for au_i in au_cols if ((au_i.lower()==f"au{int(au_idx)}_r") or (au_i.lower()==f"au0{int(au_idx)}_r"))]
                    au_cols_subset+=au_to_add
                    if len(au_to_add)==0:
                        bad_aus.append(au_idx)
                if ((len(bad_aus)>0) and (warning_printed==False)):
                    print(f"Warning - the following AUs were not included as they were not found in the dataset: {bad_aus}; valid AUs are {full_au_cols}")
                    warning_printed=True
                au_cols = au_cols_subset
            au_df = df[au_cols]

            au_data = None
            for b in range(nbins):
                df_i = au_df.iloc[int(int(config["fps"]) * (start + b) * binsize):int(
                    int(config["fps"]) * (start + b + 1) * binsize)].mean(axis=0)
                df_i.index = [i + "_t" + str((start + b) * binsize) + "-t" + str((start + b + 1) * binsize) for i in
                              list(df_i.index)]
                if type(au_data) is type(None):
                    au_data = df_i
                else:
                    au_data = pd.concat([au_data, df_i])

            all_au_data[subj] = au_data
            group_labels.append(0)
            observation_labels.append(subj)

        au_df = pd.DataFrame(all_au_data)
        au_arr = np.array(au_df.T)
        feat_names = list(au_df.index)

    else:
        group_dict = {}
        warning_printed = False
        for g, group in enumerate(selected_subgroups):
            group_dict[group] = g
            for subj in config["subgroups"][group]:
                df = pd.read_csv(config["data_directory"] + "/" + subj)
                df.columns = df.columns.str.replace(' ', '')
                au_cols = [i for i in df.columns if ((i.startswith("AU")) and (i.endswith("r")))]
                if aus_to_include!="all":
                    au_cols_subset = []
                    full_au_cols = au_cols.copy()
                    bad_aus = []
                    for au_idx in aus_to_include:
                        au_to_add = [au_i for au_i in au_cols if ((au_i.lower()==f"au{int(au_idx)}_r") or (au_i.lower()==f"au0{int(au_idx)}_r"))]
                        au_cols_subset+=au_to_add
                        if len(au_to_add)==0:
                            bad_aus.append(au_idx)
                    if ((len(bad_aus)>0) and (warning_printed==False)):
                        print(f"Warning - the following AUs were not included as they were not found in the dataset: {bad_aus}; valid AUs are {full_au_cols}")
                        warning_printed=True
                    au_cols = au_cols_subset
                au_df = df[au_cols]

                if binsize == None:
                    binsize = end - start
                    nbins = 1
                else:
                    nbins = int((end - start) / binsize)

                au_data = None
                for b in range(nbins):
                    df_i = au_df.iloc[int(int(config["fps"]) * (start + b) * binsize):int(
                        int(config["fps"]) * (start + b + 1) * binsize)].mean(axis=0)
                    df_i.index = [i + "_t" + str((start + b) * binsize) + "-t" + str((start + b + 1) * binsize) for i in
                                  list(df_i.index)]
                    if type(au_data) is type(None):
                        au_data = df_i
                    else:
                        au_data = pd.concat([au_data, df_i])

                all_au_data[subj] = au_data
                group_labels.append(g)
                observation_labels.append(subj)
            au_df = pd.DataFrame(all_au_data)
            au_arr = np.array(au_df.T)
            feat_names = list(au_df.index)

    return ActionUnits(au_arr, group_labels, observation_labels, feat_names, group_dict, None)


def BORIS_to_pose(config, verbose = True):
    """
    Intake paired BORIS one-hot-encoded observation files and pose segmented files and align them to see what behaviors
    line up with what pose modules

    :param config: the config
    :return:
    """
    boris_directory = config["boris_directory"]
    boris_to_pose_pairings = config["boris_to_pose_pairings"]
    results=None
    config_modulo = copy.deepcopy(config)
    config_modulo["project_files"]=[pairing[1] for pairing in boris_to_pose_pairings if pairing[0] is not None]
    labels_df = get_module_labels(config_modulo, 0, 1200)
    labels_flat = np.array(labels_df)
    labels_flat = [item for sublist in labels_flat for item in sublist]
    ## old way
    # modules = np.unique(labels_flat)
    # n_modules = len(modules)
    # modules = np.unique(labels_df.values.flatten())
    n_modules=int(config["n_modules"])
    modules = np.arange(0,n_modules,1)
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
            if verbose:
                print("done with " + str(pairing))
    results=results.T
    column_sums = results.sum()
    column_sums = column_sums.replace(0, 1)
    normalized_results = results / column_sums
    loss_score = np.sum(results.sum()-results.max())/np.sum(results.sum())
    if verbose:
        print(config["data_source"] + " modules map onto scored behaviors with 'loss' of " + str(loss_score))
    return results, normalized_results, loss_score

def combine_pose_modules(config, labels_df, force_no_numeric=False):
    """
    Combine pose modules based on remappings key in config

    :param config: config
    :param labels_df: from label_counter (subgroups or no_subgroups)
    :return: labels_df_remapped

    """
    try:
        n_rules=len(config["remappings"])
        print("Applying " + str(n_rules) + " remapping rules from config.")
        for r, remapping in enumerate(config["remappings"]):
            print(str(r+1) + " of " + str(n_rules) + ": Remapping " + str(remapping[0]) + " to " + str(remapping[1]))
            if remapping[0] is not None:
                for old_val in remapping[0]:
                    for column in labels_df.columns:
                        labels_df.replace(old_val, remapping[1], inplace=True)
    except:
        raise ValueError("Could not apply module remappings - check to see if something is wrong in the config.")
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

class ModuleUsage:
    def __init__(self, label_counts, group_labels, observation_labels, feat_names, group_dict, scaler):
        self.label_counts = label_counts
        self.group_labels = group_labels
        self.observation_labels = observation_labels
        self.feat_names = feat_names
        self.group_dict = group_dict
        self.scaler = scaler
        mean_check = np.allclose(np.mean(label_counts, axis=0), 0, atol=0.1)
        std_check = np.allclose(np.std(label_counts, axis=0), 1, atol=1)
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
        modules = np.unique([i.split("module")[1].split("_")[0] for i in list(df.columns) if "module" in i])
        if is_nonnum(modules[0]):
            modules = sorted(modules)
        else:
            modules = sorted([int(m) for m in modules])
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

    def module_group_comparisons_fdr(self, baseline_group, alpha=0.1):

        df = self.to_df()
        subjs = df.index
        modules = [col for col in df.columns if col != "group"]

        long_df = []
        for s, subj in enumerate(subjs):
            for mod in modules:
                long_df.append({
                    "subj_id": subj,
                    "group": df.at[subj, "group"],
                    "module": mod,
                    "value": df.at[subj, mod]
                })
        long_df = pd.DataFrame(long_df)

        formula = f"value ~ C(group, Treatment(reference='{baseline_group}')) * module"
        model = smf.mixedlm(formula, long_df, groups=long_df["subj_id"])
        result = model.fit()

        # Get model params and covariance
        coefs = result.params
        cov = result.cov_params()

        comparisons = []
        groups = long_df["group"].unique()

        # Loop over modules and groups
        for mod in modules:
            for g in groups:
                if g == baseline_group:
                    continue  # skip baseline, already in intercept

                # Construct the coefficient name for this interaction
                coef_name = f"C(group, Treatment(reference='{baseline_group}'))[T.{g}]:module[T.{mod}]"

                # If the module is the baseline module, the main effect is enough
                if f"module[T.{mod}]" not in coefs.index and coef_name not in coefs.index:
                    # baseline module
                    coef_name = f"C(group, Treatment(reference='{baseline_group}'))[T.{g}]"

                if coef_name in coefs.index:
                    est = coefs[coef_name]
                    se = np.sqrt(cov.loc[coef_name, coef_name])
                    z = est / se
                    p = 2 * (1 - stats.norm.cdf(np.abs(z)))
                else:
                    est = 0
                    se = np.nan
                    z = np.nan
                    p = np.nan

                comparisons.append({
                    "module": mod,
                    "group": g,
                    "diff_from_baseline": est,
                    "se": se,
                    "z": z,
                    "p": p
                })

        comp_df = pd.DataFrame(comparisons)

        # Apply BH FDR correction
        comp_df = comp_df.dropna(subset=["p"])
        reject, p_adj, _, _ = multipletests(comp_df["p"], alpha=alpha, method='fdr_bh')
        comp_df["p_adj"] = p_adj
        comp_df["significant"] = reject

        return comp_df, result

    def mixed_model(self):
        df = self.to_df()
        subjs = df.index
        modules = [col for col in df.columns if col != "group"]

        long_df = []
        for s, subj in enumerate(subjs):
            for mod in modules:
                long_df.append({"subj_n": s, "subj_id": subj, "group": df.at[subj, "group"], "module": mod,
                                "value": df.at[subj, mod]})

        long_df = pd.DataFrame(long_df)
        model = smf.mixedlm("value ~ group * module", long_df, groups=long_df["subj_id"])
        result = model.fit()

        return result.summary().tables[1]

    def f_oneway(self,correction="fdr_bh"):
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
        results_df["p_corr"] = multipletests(p_uncorr, method=correction)[1]
        return results_df

    def tukeyhsd(self,f_step_correction="fdr_bh",f_step_alpha=0.05,tukey_step_alpha=0.05):
        results_df = self.f_oneway(correction=f_step_correction)
        df = self.to_df()
        tukey_res_df = []
        for _, row in results_df.iterrows():
            if row["p_corr"] < f_step_alpha:
                tuk_endog = df[row["module"]].to_numpy()
                tuk_groups = df["group"].to_numpy()
                res = pairwise_tukeyhsd(endog=tuk_endog, groups=tuk_groups, alpha=tukey_step_alpha)
                res = pd.DataFrame(res._results_table.data[1:], columns=res._results_table.data[0])
                res["module"] = row["module"]
                tukey_res_df.append(res)
        try:
            tukey_res_df = pd.concat(tukey_res_df)
        except:
            tukey_res_df = None
        return tukey_res_df

    def save(self, save_path):
        if ((save_path.endswith(".pkl")==False) and (save_path.endswith(".pickle")==False)):
            save_path+=".pickle"
            print(f"Save path changed to {save_path}")
        with open(save_path, 'wb') as file:
            pickle.dump(self, file)


def get_module_usage(config, labels_df, binsize=None, modules_altered=False):
    """
    Reshape labels dataframe from label_counter_subgroups to be an array of features

    :param config: config object
    :param labels_df: labels dataframe from label_counter_subgroups
    :param binsize: width of bins in seconds; if None, no binning is performed
    :param modules_altered: must be true if modules have been remapped
    :return: object of class ModuleUsage
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
    if modules_altered:
        modules = np.unique(labels_flat)
        n_modules = len(modules)
    else:
        n_modules = int(config["n_modules"])
        modules = np.arange(0, n_modules, 1)
        modules_detect = np.unique(labels_flat)
        for m in modules_detect:
            if is_nonnum(m):
                print("Warning, detected non-numeric modules - consider setting 'modules_altered' to True (False by defualt)")
                break

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
        std_check = np.allclose(np.std(transition_counts, axis=0), 1, atol=1)
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

    def save(self, save_path):
        if ((save_path.endswith(".pkl")==False) and (save_path.endswith(".pickle")==False)):
            save_path+=".pickle"
            print(f"Save path changed to {save_path}")
        with open(save_path, 'wb') as file:
            pickle.dump(self, file)

def get_module_transitions(config, labels_df, modules_altered=False):
    """
    Reshape labels dataframe from label_counter_subgroups to be an array of features

    :param config: config object
    :param labels_df: labels dataframe from label_counter_subgroups
    :param binsize: width of bins in seconds; if None, no binning is performed
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
    if modules_altered:
        modules = np.unique(labels_flat)
        n_modules = len(modules)
    else:
        n_modules = int(config["n_modules"])
        modules = np.arange(0, n_modules, 1)
        modules_detect = np.unique(labels_flat)
        for m in modules_detect:
            if is_nonnum(m):
                print("Warning, detected non-numeric modules - consider setting 'modules_altered' to True (False by defualt)")
                break

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

def load_module_feature_object(module_feature_object_path):
    with open(module_feature_object_path, 'rb') as file:
        module_feature_object = pickle.load(file)
    return module_feature_object

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
    elif module_feature_object.__class__.__name__=="KeypointFeature":
        X=module_feature_object.keypoint_feature
        y=module_feature_object.group_labels
        obj_type="keypoint feature"
    elif module_feature_object.__class__.__name__=="ActionUnits":
        X=module_feature_object.action_units
        y=module_feature_object.group_labels
        obj_type="action units"
    else:
        raise ValueError(f'module_feature_object class must be ModuleUsage or ModuleTransitions or KeypointFeature, not {module_feature_object.__class__}')

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
        X=module_feature_object.label_counts
        y=module_feature_object.group_labels
        obj_type="module usage"
    elif module_feature_object.__class__.__name__=="ModuleTransitions":
        X=module_feature_object.transition_counts
        y=module_feature_object.group_labels
        obj_type="module transitions"
    elif module_feature_object.__class__.__name__=="KeypointFeature":
        X=module_feature_object.keypoint_feature
        y=module_feature_object.group_labels
        obj_type="keypoint feature"
    elif module_feature_object.__class__.__name__=="ActionUnits":
        X=module_feature_object.action_units
        y=module_feature_object.group_labels
        obj_type="action units"
    else:
        raise ValueError(f'module_feature_object class must be ModuleUsage or ModuleTransitions or KeypointFeature, not {module_feature_object.__class__}')

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
        X=module_feature_object.label_counts
        y=module_feature_object.group_labels
        obj_type="module usage"
    elif module_feature_object.__class__.__name__=="ModuleTransitions":
        X=module_feature_object.transition_counts
        y=module_feature_object.group_labels
        obj_type="module transitions"
    elif module_feature_object.__class__.__name__=="KeypointFeature":
        X=module_feature_object.keypoint_feature
        y=module_feature_object.group_labels
        obj_type="keypoint feature"
    elif module_feature_object.__class__.__name__=="ActionUnits":
        X=module_feature_object.action_units
        y=module_feature_object.group_labels
        obj_type="action units"
    else:
        raise ValueError(f'module_feature_object class must be ModuleUsage or ModuleTransitions or KeypointFeature, not {module_feature_object.__class__}')
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

def get_distance(module_feature_object, method="euclidean"):
    """
    Get distance

    :param module_feature_object:
    :param method:
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
    elif module_feature_object.__class__.__name__=="KeypointFeature":
        X=module_feature_object.keypoint_feature
        y=module_feature_object.group_labels
        obj_type="keypoint feature"
    elif module_feature_object.__class__.__name__=="ActionUnits":
        X=module_feature_object.action_units
        y=module_feature_object.group_labels
        obj_type="keypoint feature"
    else:
        raise ValueError(f'module_feature_object class must be ModuleUsage or ModuleTransitions or KeypointFeature, not {module_feature_object.__class__}')
    centroids = np.zeros([len(np.unique(X)), len(module_feature_object.feat_names)])
    for group in sorted(np.unique(module_feature_object.group_labels)):
        centroids[group, :] = np.mean(X[np.array(module_feature_object.group_labels) == group], axis=0)

    dist_mat = np.zeros([len(np.unique(module_feature_object.group_labels)), len(module_feature_object.group_labels)])
    for o, obs in enumerate(X):
        for group in sorted(np.unique(module_feature_object.group_labels)):
            if method == "euclidean":
                dist_mat[group, o] = distance.euclidean(obs, centroids[group, :])
            elif method == "braycurtis":
                dist_mat[group, o] = distance.braycurtis(obs, centroids[group, :])
            elif method == "cityblock":
                dist_mat[group, o] = distance.cityblock(obs, centroids[group, :])
            elif method == "correlation":
                dist_mat[group, o] = distance.correlation(obs, centroids[group, :])

    return dist_mat.T

def regress(module_feature_object, dose_dict, method="LinearRegression", degree=1, alpha = 1):
    """
    Regress a continuous variable (e.g., dose, stimulus value) from a module or keypoint feature object

    :param module_feature_object: ModuleUsage, ModuleTransitions, or KeypointFeature object
    :param dose_dict: dictionary with keys corresponding to subgroups and items corresponding to variable
    :param method: method of regression; either "LinearRegression" (default) or "Ridge" or "Lasso"
    :param degree: polynomial degree; default 1
    :param alpha: alpha (default 1; only applies for regularized regression, i.e. Ridge and Lasso)
    :return: regression model, dose_labels

    """
    if module_feature_object.__class__.__name__=="ModuleUsage":
        X=module_feature_object.label_counts
    elif module_feature_object.__class__.__name__=="ModuleTransitions":
        X=module_feature_object.transition_counts
    elif module_feature_object.__class__.__name__=="KeypointFeature":
        X=module_feature_object.keypoint_feature
    else:
        raise ValueError(f'module_feature_object class must be ModuleUsage or ModuleTransitions or KeypointFeature, not {module_feature_object.__class__}')
    reverse_grp_dict = {v: k for k, v in module_feature_object.group_dict.items()}
    dose_labels = [dose_dict[reverse_grp_dict[i]] for i in module_feature_object.group_labels]
    if ((degree<=0) or (degree%1!=0)):
        raise ValueError("Degree must be a positive, non-zero integer!")
    if degree==1:
        if method=="LinearRegression":
            reg = LinearRegression().fit(X, dose_labels)
        elif method=="Lasso":
            reg = Lasso(alpha=alpha).fit(X, dose_labels)
        elif method=="Ridge":
            reg = Ridge(alpha=alpha).fit(X, dose_labels)
    elif degree>1:
        if method=="LinearRegression":
            reg = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression())
            reg.fit(X, dose_labels)
        elif method=="Lasso":
            reg = make_pipeline(PolynomialFeatures(degree=degree), Lasso(alpha=alpha))
            reg.fit(X, dose_labels)
        elif method=="Ridge":
            reg = make_pipeline(PolynomialFeatures(degree=degree), Ridge(alpha=alpha))
            reg.fit(X, dose_labels)

    return reg, dose_labels

def loocv_regression(module_feature_object, dose_dict, method="LinearRegression", constrain_pos = True, degree=1, alpha=1):
    """
    Perform LOOCV for linear regression

    :param module_feature_object: ModuleUsage, ModuleTransitions, or KeypointFeature object
    :param dose_dict: dictionary with keys corresponding to subgroups and items corresponding to variable
    :param method: method of regression; either "LinearRegression" (default) or "Ridge" or "Lasso"
    :param constrain_pos: constrain to only positive values; true by default
    :param degree: degree of polynomial (default 1)
    :param alpha: alpha (default 1; only applies for regularized regression, i.e. Ridge and Lasso)
    :return: held-out predictions, squared error
    """
    loocv_preds = []
    if module_feature_object.__class__.__name__=="ModuleUsage":
        X=module_feature_object.label_counts
        y=module_feature_object.group_labels
        obj_type="module usage"
    elif module_feature_object.__class__.__name__=="ModuleTransitions":
        X=module_feature_object.transition_counts
        y=module_feature_object.group_labels
        obj_type="module transitions"
    elif module_feature_object.__class__.__name__=="KeypointFeature":
        X=module_feature_object.keypoint_feature
        y=module_feature_object.group_labels
        obj_type="keypoint feature"
    else:
        raise ValueError(f'module_feature_object class must be ModuleUsage or ModuleTransitions or KeypointFeature, not {module_feature_object.__class__}')
    reverse_grp_dict = {v: k for k, v in module_feature_object.group_dict.items()}
    y = [dose_dict[reverse_grp_dict[i]] for i in module_feature_object.group_labels]
    for obs in range(len(y)):
        sub_X = np.delete(X, obs, axis=0)
        sub_y = y[:obs] + y[obs + 1:]
        X_i = X[obs, :]
        y_i = y[obs]
        if ((degree<=0) or (degree%1!=0)):
            raise ValueError("Degree must be a positive, non-zero integer!")
        if degree==1:
            if method=="LinearRegression":
                reg = LinearRegression().fit(sub_X, sub_y)
            elif method=="Lasso":
                reg = Lasso(alpha=alpha).fit(sub_X, sub_y)
            elif method=="Ridge":
                reg = Ridge(alpha=alpha).fit(sub_X, sub_y)
        elif degree>1:
            if method=="LinearRegression":
                reg = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression())
                reg.fit(sub_X, sub_y)
            elif method=="Lasso":
                reg = make_pipeline(PolynomialFeatures(degree=degree), Lasso(alpha=alpha))
                reg.fit(sub_X, sub_y)
            elif method=="Ridge":
                reg = make_pipeline(PolynomialFeatures(degree=degree), Ridge(alpha=alpha))
                reg.fit(sub_X, sub_y)

        loocv_preds.append(reg.predict(X_i.reshape(1, -1))[0])
    loocv_preds = np.array(loocv_preds)
    if constrain_pos:
        loocv_preds[loocv_preds<0]=0
    y = np.array(y)
    sq_err = np.square(loocv_preds-y)
    return loocv_preds, sq_err