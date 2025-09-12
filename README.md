# MARIPoSA
<img src="other/MARIPoSA_icon.png" alt="alt text" width="150" height="150">

### Description
MARIPoSA (Manageable And Reproducible Integrated Pose Segmentation Analysis) is a platform for unified analysis of naturalistic animal and human behavior data in the native formats of a variety of open source tools. Computational pose analysis represents a promising future direction in both animal and human research; however, tools for investigation in this area generally output raw data in non-interchangeable formats, leaving researchers without a clear direction for interpreting their data and comparing with what exists in the literature. The purpose of MARIPoSA is to provide a resource for analyzing and visualizing naturalistic behavioral data while also promoting the creation of robust shared technical practices for the field at large.

### Installation
Installing the conda environment:

    conda env create -f MARIPoSA/conda/environment.yaml

### Using the GUI
Activate the newly installed conda environment:

    conda activate mariposa

Run the GUI:

    python MARIPoSA/main.py

In the GUI:
1. Click "New project"
2. Enter project metadata. 
   - NOTE: For "Path to data directory" - if you datatype is B-SOiD or Keypoints-MoSeq, you should enter the path to a folder containing all the label .csv files with their original names; if your datatype is VAME you should enter the path to the "results" folder ouptut by VAME which contains nested subdirectories of a structure like "video1/VAME/hmm-15/15_km_label_video1.npy"
4. Start your analysis!


### Using the Python Commands
Activate the newly installed conda environment 

    conda activate mariposa

And you are ready to start analyzing some data. 

### Learning to use MARIPoSA
The MARIPoSA python command utilities can be learned from three main resources:
1. Reading the function docstrings
2. The console log window of the GUI, which records the functions being executed in the GUI and can be exported to a text file for not only reviewing executed analyses but also learning the python command structure
3. The Jupyter notebook demos!

There exist 7 Jupyter notebook demos, labeled "PS" or "PE" based on whether they apply to pose segmentation or estimation data. The notebooks can be found in the "demo" folder of the Github repository or in the documentation website, and their contents are as follows:

* **Demo 1 (PS/PE): Project Creation**
  * 1.0 Import and Setup
  * 1.1: Creating a pose segmentation project
  * 1.2 Creating a pose estimation project
* **Demo 2 (PS): Visualization of Pose Module Usage**
  * 2.1: Loading data and simply visualizing usage (no subgroups or time binning)
  * 2.2: Loading data and visualizing usage by subgroups
* **Demo 3 (PS): Embedding of Pose Segmentation Data**
  * 3.1: Embedding with PCA
  * 3.2 Embedding with LDA
* **Demo 4 (PS): Regression and Classification of Group Conditions in Pose Segmentation Data**
  * 4.1: Classification
  * 4.2: Regression
* **Demo 5 (PS): Comparing Pose Modules to Manual Scoring Data**
  * 5.1: Remapping pose modules to grouped classes
  * 5.2: Incorporating manual scoring data from BORIS to remap modules
* **Demo 6 (PS): Simulating Pose Segmentation Data**
  * 6.1: Simulating time-series categorical pose segmentation data
  * 6.2: Simulating module usage
* **Demo 7 (PE): Analyzing Keypoint Displacement in Pose Estimation Data**
  * 7.1: Measuring and plotting keypoint travel in animals
  * 7.2 Using keypoint travel in animals for embedding and classification
* **Demo 8 (PE): Analyzing Keypoint Displacement and Action Units in human facial pose data from OpenFace**
  * 8.0: Getting familiar with OpenFace data
  * 8.1 Measuring and Analyzing Keypoint Travel in OpenFace Data
  * 8.2 Working with Action Units

