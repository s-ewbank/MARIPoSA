import os
from datetime import datetime
import yaml
import subprocess
import sys
import pandas as pd
import numpy as np
import h5py

def create_PS_project(project_name,data_directory,data_source,output_directory,fps,n_modules):
    """
    Make project directory and write config_PS.yaml file.

    :param project_name: what to call your creation
    :param data_directory: path to source data
    :param data_source: B-SOiD, VAME, or Keypoint-MoSeq
    :param output_directory: path where output directory and config file should be created
    :param fps: frames per second
    :param n_modules: number of pose states / modules
    :return: path to saved config (in specified output directory)
    """
    if data_source == "B-SOiD":
        project_files = sorted(os.listdir(data_directory))
    elif data_source == "VAME":
        project_files = sorted([item for item in os.listdir(data_directory) if
                                (os.path.isdir(os.path.join(data_directory, item)) and (item!="community_cohort"))])
    elif data_source == "Keypoint-MoSeq":
        project_files = sorted(os.listdir(data_directory))
    project_directory = str(output_directory+"/"+datetime.now().strftime('%y%m%d_')+project_name)
    if os.path.exists(project_directory):
        print(f"[{datetime.now()}] Warning - the path '{project_directory}' exists - a config will still be made (possible overwrite).")
    else:
        os.mkdir(project_directory)

    PS_config = {}
    PS_config["project_name"] = datetime.now().strftime('%y%m%d_') + project_name
    PS_config["data_directory"] = str(data_directory)
    PS_config["project_directory"] = str(output_directory+"/"+datetime.now().strftime('%y%m%d_')+project_name)
    PS_config["data_type"] = "Pose segmentation"
    PS_config["data_source"] = str(data_source)
    PS_config["n_modules"] = str(int(n_modules))
    PS_config["fps"] = str(fps)
    if data_source == "VAME":
        vame_models = sorted(os.listdir(PS_config["data_directory"]+"/"+project_files[0]+"/VAME/"))
        compatible_vame_models = [i for i in vame_models if (i.endswith(f"-{PS_config['n_modules']}") or i.endswith(f"-{PS_config['n_modules']}/"))]
        if len(compatible_vame_models)==0:
            raise ValueError(f"Specified number of states, {str(int(n_modules))}, not compatible with VAME models found: {vame_models}")
        else:
            PS_config["vame_model_name"] = compatible_vame_models[0]
    PS_config["project_files"] = project_files
    PS_config["subgroups"] = {"group1" : project_files}
    PS_config["remappings"] = [["#old_poses; e.g., [1,2,3]","#new_pose; e.g., 400"],
                          ["#old_poses; e.g., [1,2,3]","#new_pose; e.g., 400"]]
    PS_config["boris_directory"] = "#/path/to/boris"
    PS_config["boris_to_pose_pairings"]=[["#boris file 1 here","#pose file 1 here"],
                                      ["#boris file 2 here","#pose file 2 here"],
                                      ["#boris file n here","#pose file n here"]]
    with open(project_directory+"/config_PS.yaml", "w") as outfile:
        yaml.Dumper.ignore_aliases = lambda self, data: True
        yaml.dump(PS_config, outfile, default_flow_style=False, sort_keys=False, Dumper=yaml.Dumper)

    with open(project_directory+"/config_PS.yaml", 'r') as file:
        filedata = file.read()
    filedata = filedata.replace("'#", "#")
    filedata = filedata.replace('"#', "#")
    with open(project_directory+"/config_PS.yaml", 'w') as file:
        file.write(filedata)

    print(f"[{datetime.now()}] Made project {PS_config['project_name']} and saved at {project_directory+'/config_PS.yaml'}")
    return project_directory+'/config_PS.yaml'

def create_PE_project(project_name,data_directory,data_source,output_directory,fps):
    """
    Make project directory and write config_PE.yaml file.

    :param project_name: what to call your creation
    :param data_directory: path to source data
    :param data_source: DeepLabCut or SLEAP
    :param output_directory: path where output directory and config file should be created
    :param fps: frames per second
    :return: path to saved config (in specified output directory)
    """
    if data_source=="DeepLabCut":
        project_files=[i for i in sorted(os.listdir(data_directory)) if i.endswith(".csv")]
        test_df = pd.read_csv(data_directory + "/" + project_files[0], header=[0, 1, 2])
        keypoints = [str(i) for i in np.unique([i[1] for i in test_df.columns]) if str(i)!="bodyparts"]
    elif data_source=="SLEAP":
        prediction_files=sorted(os.listdir(data_directory))
        project_files=[]
        for file in prediction_files:
            with h5py.File(data_directory+"/"+file, "r") as f:
                node_names = f['node_names'][:]
                node_names = [node.decode('utf-8') for node in node_names]
                track_names = f['track_names'][:]
                track_names = [tr.decode('utf-8') for tr in track_names]
                for tr in track_names:
                    project_files.append(file+":::"+tr)
        keypoints=node_names
    elif data_source=="OpenFace":
        project_files=[i for i in sorted(os.listdir(data_directory)) if i.endswith(".csv")]
        test_df = pd.read_csv(data_directory+"/"+project_files[0])
        test_df.columns = test_df.columns.str.replace(' ', '')
        y_coords = [i for i in test_df.columns if i.lower().startswith("y_")]
        keypoints = ["kp_" + i[i.lower().index("y_") + 2:] for i in y_coords]
    project_directory=str(output_directory+"/"+datetime.now().strftime('%y%m%d_')+project_name)
    if os.path.exists(project_directory):
        print(f"[{datetime.now()}] Warning - the path '{project_directory}' exists - a config will still be made (possible overwrite).")
    else:
        os.mkdir(project_directory)

    PE_config = {}
    PE_config["project_name"]=datetime.now().strftime('%y%m%d_') + project_name
    PE_config["data_directory"]=str(data_directory)
    PE_config["project_directory"]=str(output_directory+"/"+datetime.now().strftime('%y%m%d_')+project_name)
    PE_config["data_type"]="Pose estimation"
    PE_config["data_source"]=str(data_source)
    PE_config["fps"]=str(fps)
    PE_config["keypoints"]=keypoints
    PE_config["project_files"]=project_files
    PE_config["subgroups"]={"group1" : project_files}
    with open(project_directory+"/config_PE.yaml", "w") as outfile:
        yaml.Dumper.ignore_aliases = lambda self, data: True
        yaml.dump(PE_config, outfile, default_flow_style=False, sort_keys=False, Dumper=yaml.Dumper)

    print(f"[{datetime.now()}] Made project {PE_config['project_name']} and saved at {project_directory+'/config_PE.yaml'}")
    return project_directory+'/config_PE.yaml'

def load_project(config_path):
    """
    Loads project from config_PS.yaml or config_PE.yaml file

    :param config_path: path to config file:
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def save_edited_project(config, config_path):
    """
    Loads project from config_PS.yaml or config_PE.yaml file

    :param config_path: path to config file:
    """
    if os.path.exists(config_path):
        existed=True
    else:
        existed=False

    with open(config_path, "w") as outfile:
        yaml.Dumper.ignore_aliases = lambda self, data: True
        yaml.dump(config, outfile, default_flow_style=False, sort_keys=False, Dumper=yaml.Dumper)

    with open(config_path, 'r') as file:
        filedata = file.read()
    filedata = filedata.replace("'#", "#")
    filedata = filedata.replace('"#', "#")
    with open(config_path, 'w') as file:
        file.write(filedata)

    if existed:
        print(f"[{datetime.now()}] Saved specified config object at {config_path}, overwriting an existing file at that path")
    else:
        print(f"[{datetime.now()}] Saved specified config object at {config_path}")

def edit_config(config_path):
    config = yaml.safe_load(config_path)
    if sys.platform=="win32":
        os.startfile(config_path)
    elif sys.platform=="darwin":
        subprocess.call(["open",config_path])
    else:
        subprocess.call(["xdg-open", config_path])
