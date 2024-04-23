# MARIPoSA

### Description
A platform for Manageable And Reproducible Integrated Pose Segmentation Analysis (MARIPoSA).

### Installation
Installing the conda environment:

    conda env create -f MARIPOSA/conda/environment.yaml

### Using the GUI
Activate the newly installed conda environment:

    conda activate mariposa

Run the GUI:

    python main.py

In the GUI:
1. Click "New project"
2. Enter project metadata. 
   - NOTE: For "Path to data directory" - if you datatype is B-SOiD or Keypoints-MoSeq, you should enter the path to a folder containing all the label .csv files with their original names; if your datatype is VAME you should enter the path to the "results" folder ouptut by VAME which contains nested subdirectories of a structure like "video1/VAME/hmm-15/15_km_label_video1.npy"
4. Start your analysis!


### Using the Python Commands
Activate the newly installed conda environment 

    conda activate mariposa

Check out the demo scripts for a full analysis and plotting pipeline!