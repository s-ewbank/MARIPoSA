import os
from datetime import datetime
import yaml
import subprocess
import sys

def create_project(project_name,data_directory,datatype,output_directory,fps):
    """
    Make project directory and write project_info.py file.
    :param project_name: what to call your creation
    :param data_directory: path to source data
    :param datatype: B-SOiD or VAME
    :param output_directory: path where PoseVis output should be created
    """
    project_directory=str(output_directory+"/"+datetime.now().strftime('%y%m%d_')+project_name)
    if datatype=="B-SOiD":
        project_files="".join(["\n  - '"+i+"'" for i in sorted(os.listdir(data_directory)) if ".csv" in i])
        project_files_double_indent = "".join(["\n    - '" + i + "'" for i in sorted(os.listdir(data_directory)) if ".csv" in i])
    elif datatype=="VAME":
        project_file_list = sorted([item for item in os.listdir(data_directory) if os.path.isdir(os.path.join(data_directory, item))])
        project_files = "".join(["\n  - '" + i + "'" for i in project_file_list])
        project_files_double_indent = "".join(["\n  - '" + i + "'" for i in project_file_list])
    elif datatype=="Keypoint-MoSeq":
        project_files="".join(["\n  - '"+i+"'" for i in sorted(os.listdir(data_directory)) if ".csv" in i])
        project_files_double_indent = "".join(["\n    - '" + i + "'" for i in sorted(os.listdir(data_directory)) if ".csv" in i])
    os.mkdir(project_directory)
    f = open(project_directory+"/config.yaml", "w")
    f.write("project_name: '" + datetime.now().strftime('%y%m%d_') + project_name + "'")
    f.write("\n\ndata_directory: '" + str(data_directory) + "'")
    f.write("\n\nproject_directory: '" + str(project_directory) + "'")
    f.write("\n\ndata_type: '" + str(datatype) + "'")
    f.write("\n\nfps: " + str(fps))
    f.write("\n\nproject_files: " + project_files)
    f.write("\n\nsubgroups:")
    f.write("\n  group1:" + project_files_double_indent)
    f.write("\n\nremappings:")
    f.write("\n  - - #old_poses; e.g., [1,2,3]")
    f.write("\n    - #new_pose; e.g., 400")
    f.write("\n  - - #old_poses; e.g., [1,2,3]")
    f.write("\n    - #new_pose; e.g., 400")
    f.write("\n\nboris_directory: #boris_directory")
    f.write("\n\nboris_to_pose_pairings:")
    f.write("\n  - - #boris file 1 here")
    f.write("\n    - #pose file 1 here")
    f.write("\n  - - #boris file 2 here")
    f.write("\n    - #pose file 2 here")
    f.write("\n  - - #boris file n here")
    f.write("\n    - #pose file n here")
    f.close()

def make_dlc_config(dlc_path, fps=None, config=None, yaml_path=None):
    """
    Create a DLC config
    :param path:
    :param fps: frames per second (must be specified if no config is given)
    :param config: pose config
    :param yaml_path:
    :return:
    """
    DLC_config = {}
    DLC_config["path"] = dlc_path
    if fps is None:
        if config is not None:
            DLC_config["fps"] = config["fps"]
        else:
            raise ValueError("You need to have either config or fps specified (or both).")
    elif ((config is None) or (config["fps"] == fps)):
        DLC_config["fps"] = fps
    DLC_files = sorted(os.listdir(DLC_config["path"]))

    DLC_config["subgroups"] = {}

    if config is not None:
        for subgroup in config["subgroups"].keys():
            DLC_subgroup = []
            for file in DLC_files:
                end_handle = file.split("DLC_")[-1]
                file_handle = file.split("DLC_" + end_handle)[0]
                if len([i for i in config["subgroups"][subgroup] if file_handle in i]) > 0:
                    DLC_subgroup.append(file)
            DLC_config["subgroups"][subgroup] = DLC_subgroup

    if config is not None:
        yaml_path = config["project_directory"]
    with open(yaml_path + '/DLC_config.yaml', 'w') as outfile:
        yaml.dump(DLC_config, outfile, default_flow_style=False)

    return DLC_config

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
