.. MARIPoSA documentation master file, created by
   sphinx-quickstart on Mon Sep 30 14:15:53 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to MARIPoSA's documentation!
====================================

.. image:: ../other/MARIPoSA_icon.png
   :width: 100px
   :alt: MARIPoSA

A platform for Manageable And Reproducible Integrated Pose Segmentation Analysis (MARIPoSA). See below for instructions on a quick start, or navigate through the tutorials and function descriptions for more detailed descriptions of the software functionality.

Quick Start
-------------------------------------

With GUI:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
First, install the conda environment:

.. code-block:: bash

    conda env create -f MARIPoSA/conda/environment.yaml

Activate the newly installed conda environment:

.. code-block:: bash

    conda activate mariposa

Run the GUI:

.. code-block:: bash

    python MARIPoSA/main.py

In the GUI:

1. Click "New project"

2. Enter project metadata.

   - NOTE: For "Path to data directory" - if you datatype is B-SOiD or Keypoints-MoSeq, you should enter the path to a folder containing all the label .csv files with their original names; if your datatype is VAME you should enter the path to the "results" folder ouptut by VAME which contains nested subdirectories of a structure like "video1/VAME/hmm-15/15_km_label_video1.npy"

3. Start your analysis!

With Python package:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Activate the newly installed conda environment 

.. code-block:: bash

    conda activate mariposa

Check out the demo scripts for a full analysis and plotting pipeline!

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   demo1_start
   demo2_dlc
   demo3_pca
   demo4_pk
   metadata
   analysis
   plot
