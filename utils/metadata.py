import os
from datetime import datetime
import yaml
import subprocess
import sys

def create_PS_project(project_name,data_directory,data_source,output_directory,fps):
    """
    Make project directory and write project_info.py file.
    :param project_name: what to call your creation
    :param data_directory: path to source data
    :param data_source: B-SOiD, VAME, or Keypoint-MoSeq
    :param output_directory: path where PoseVis output should be created
    """

    if data_source == "B-SOiD":
        project_files = sorted(os.listdir(data_directory))
    elif data_source == "VAME":
        project_files = sorted([item for item in os.listdir(data_directory) if os.path.isdir(os.path.join(data_directory, item))])
    elif data_source == "Keypoint-MoSeq":
        project_files = sorted(os.listdir(data_directory))
    project_directory = str(output_directory+"/"+datetime.now().strftime('%y%m%d_')+project_name)
    os.mkdir(project_directory)

    PS_config = {}
    PS_config["project_name"] = datetime.now().strftime('%y%m%d_') + project_name
    PS_config["data_directory"] = str(data_directory)
    PS_config["project_directory"] = str(output_directory+"/"+datetime.now().strftime('%y%m%d_')+project_name)
    PS_config["data_type"] = "Pose segmentation"
    PS_config["data_source"] = str(data_source)
    PS_config["fps"] = str(fps)
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

    # Fix comments
    with open(project_directory+"/config_PS.yaml", 'r') as file:
        filedata = file.read()
    filedata = filedata.replace("'#", "#")
    filedata = filedata.replace('"#', "#")
    with open(project_directory+"/config_PS.yaml", 'w') as file:
        file.write(filedata)

def create_PE_project(project_name,data_directory,data_source,output_directory,fps):
    """
    Make project directory and write project_info.py file.
    :param project_name: what to call your creation
    :param data_directory: path to source data
    :param datatype: DeepLabCut or SLEAP
    :param output_directory: path where PoseVis output should be created
    """
    if data_source=="DeepLabCut":
        project_files=[i for i in sorted(os.listdir(data_directory)) if i.endswith(".csv")]
    elif data_source=="SLEAP":
        project_files=sorted(os.listdir(data_directory))
    project_directory=str(output_directory+"/"+datetime.now().strftime('%y%m%d_')+project_name)
    os.mkdir(project_directory)

    PE_config = {}
    PE_config["project_name"]=datetime.now().strftime('%y%m%d_') + project_name
    PE_config["data_directory"]=str(data_directory)
    PE_config["project_directory"]=str(output_directory+"/"+datetime.now().strftime('%y%m%d_')+project_name)
    PE_config["data_type"]="Pose estimation"
    PE_config["data_source"]=str(data_source)
    PE_config["fps"]=str(fps)
    PE_config["project_files"]=project_files
    PE_config["subgroups"]={"group1" : project_files}
    with open(project_directory+"/config_PE.yaml", "w") as outfile:
        yaml.Dumper.ignore_aliases = lambda self, data: True
        yaml.dump(PE_config, outfile, default_flow_style=False, sort_keys=False, Dumper=yaml.Dumper)

def load_project(config_path):
    """
    loads project from config.yaml file
    :param config_path: path to config.yaml file:
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def edit_config(config_path):
    config = yaml.safe_load(config_path)
    if sys.platform=="win32":
        os.startfile(config_path)
    elif sys.platform=="darwin":
        subprocess.call(["open",config_path])
    else:
        subprocess.call(["xdg-open", config_path])
