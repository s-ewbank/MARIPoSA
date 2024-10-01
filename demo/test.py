#set up paths
import os
import sys
import matplotlib.pyplot as plt
import importlib
import pandas as pd
import numpy as np

from sklearn.tree import plot_tree
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression as LR
import scipy

#This block only important for running as script
script_dir = os.path.dirname(os.path.abspath(__file__))
mariposa_dir = os.path.dirname(script_dir)
utils_dir = os.path.join(mariposa_dir, 'utils')
sys.path.append(utils_dir)
sys.path.append(mariposa_dir)

#import utils
from utils import analysis, plot, metadata

importlib.reload(analysis)
importlib.reload(plot)

config_path="/Users/snewbank/Behavior/MARIPOSA_test/240901_KPMS-full/config_PS.yaml"

config = metadata.load_project(config_path)

labels_df, n_modules = analysis.label_counter_subgroups(config, 0, 1200)

binsizes=[30,60,120,300,600,1200]
n_features=[5,10,15,20,25,30,35,40,80]
test_matrix=np.zeros([len(binsizes),len(n_features)])
for b,binsize in enumerate(binsizes):
    for f,feats in enumerate(n_features):
        lda_result = analysis.lda_labels_timebins(config,labels_df,binsize,select_features=feats,loocv=True)
        print(f"Accuracy for binsize={binsize} and nfeats={feats} is {lda_result.loocv_accuracy}")
        test_matrix[b,f]=lda_result.loocv_accuracy


plt.figure(figsize=(4,2))
plt.imshow(test_matrix,cmap="Greens")
plt.xticks(range(len(n_features)),n_features)
plt.xlabel("Number of PCA-selected features")
plt.yticks(range(len(binsizes)),binsizes)
plt.ylabel("Bin size (s)")
plt.colorbar()
plt.tight_layout()
plt.savefig("/Users/snewbank/Behavior/MARIPOSA_test/tests/kpms-bin-vs-feat-var.png",dpi=500)


# fig1 = plot.plot_lda(config, lda_result, cmap="viridis_r")
    # fig2 = plot.plot_conf_mat(lda_result)
    #
    # print(f"Accuracy for n_features={n_features} is {lda_result.loocv_accuracy}")
