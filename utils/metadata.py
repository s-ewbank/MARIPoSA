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
    f.close()


def load_project(config_path):
    """
    loads project from project_info.py file
    :param project_info_file:
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
