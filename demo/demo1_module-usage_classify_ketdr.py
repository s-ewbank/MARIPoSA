#set up paths
import os
import sys
import matplotlib.pyplot as plt
import importlib
import pandas as pd
import numpy as np

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

model1 = "/Users/snewbank/Behavior/MARIPOSA_test/240730_vame-all/"
model2 = "/Users/snewbank/Behavior/MARIPOSA_test/240605_240605_BSOID-test/"
model3 = "/Users/snewbank/Behavior/MARIPOSA_test/240605_kpms/"

for model in [model1, model2, model3]:

    config_path = model+"/config.yaml"
    save_path = model+"/demo/"
    save=True

    #Load config
    config = metadata.load_project(config_path)

    #plot without subgroups
    # labels_df, n_modules = analysis.label_counter_nosubgroups(config,0,1200)
    # fig = plot.plot_module_usage(config, labels_df, 0, 1200,style="points")
    # if save == True:
    #     plt.savefig(save_path+"labels_no_subgroups.png",dpi=500)

    #plot with subgroups
    labels_df, n_modules = analysis.label_counter_subgroups(config,0,1200)
    fig = plot.plot_module_usage_subgroups(config, labels_df, 0, 1200,figW=5,figH=2.5,legend_pos="outside")
    if save == True:
        plt.savefig(save_path+"labels_subgroups.png",dpi=500)

    #network comparison plot
    comparison_groups=["sal","k10"]
    labels_df, n_modules = analysis.label_counter_subgroups(config,0,1200)
    fig = plot.network_pairwise_comparison(config, labels_df, 0, 1200, comparison_groups,cmap="PRGn")
    if save == True:
        plt.savefig(save_path+"network_comparison.png",dpi=500)

    #plot over time
    group="k10"
    labels_df, n_modules = analysis.label_counter_subgroups(config,0,60*50,selected_subgroups=[group])
    fig = plot.SandPlotClusterFrequency_OverTime(config, labels_df[group],0, 60, 50)
    if save==True:
        plt.savefig(save_path+"sandplot.png",dpi=500)

    # Linear discriminant analysis - scan
    binsizes=[0.5*60,1*60,2.5*60,5*60,10*60,20*60]
    accuracies=[]
    for binsize in binsizes:
        labels_df, n_modules = analysis.label_counter_subgroups(config,0,1200)
        lda, lda_embeddings, label_counts, group_labels, group_dict, nbins = analysis.lda_labels_timebins(config,labels_df,binsize)
        confusion, class_num, class_labels, accuracy = analysis.loocv_conf_mat(lda, label_counts, group_labels, group_dict)
        accuracies.append(accuracy)

    loocv_df = pd.DataFrame({"binsize": binsizes, "LOOCV": accuracies})
    best_bin_arg=np.argmax(loocv_df.LOOCV)
    best_bin=loocv_df.binsize[best_bin_arg]
    fig = plt.figure(figsize=(4,2.5))
    plt.plot(loocv_df.binsize,loocv_df.LOOCV,color="black",marker="o")
    plt.xlabel("Bin size (s)")
    plt.ylabel("Linear Discriminant Analysis \nLOOCV Accuracy")
    fig.tight_layout()
    if save==True:
        plt.savefig(save_path+"lda_loocv_scan.png",dpi=500)
        loocv_df.to_csv(save_path+"lda_loocv.csv")

    # Linear discriminant analysis - plot
    binsize=best_bin
    labels_df, n_modules = analysis.label_counter_subgroups(config,0,1200)
    lda, lda_embeddings, label_counts, group_labels, group_dict, nbins = analysis.lda_labels_timebins(config,labels_df,binsize)
    fig = plot.plot_lda(config, lda, lda_embeddings, group_labels, nbins, binsize, cmap="viridis_r")
    plt.savefig(save_path+"lda_embeddings.png",dpi=500)
    confusion, class_num, class_labels, accuracy = analysis.loocv_conf_mat(lda, label_counts, group_labels, group_dict)
    plot.plot_conf_mat(confusion, class_num, class_labels,alt_title="Linear Discriminant Analysis\nConfusion Matrix")
    if save == True:
        plt.savefig(save_path+"lda_confmat.png",dpi=500)

    #Logistic regression analysis - scan
    binsizes=[0.5*60,1*60,2.5*60,5*60,10*60,20*60]
    accuracies=[]
    for binsize in binsizes:
        labels_df, n_modules = analysis.label_counter_subgroups(config,0,1200)
        lr, group_labels, label_counts, group_dict, nbins = analysis.lr_labels_timebins(config, labels_df, binsize)
        confusion, class_num, class_labels, accuracy = analysis.loocv_conf_mat(lr, label_counts, group_labels, group_dict)
        accuracies.append(accuracy)


    loocv_df = pd.DataFrame({"binsize": binsizes, "LOOCV": accuracies})
    best_bin_arg=np.argmax(loocv_df.LOOCV)
    best_bin=loocv_df.binsize[best_bin_arg]
    fig = plt.figure(figsize=(4,2.5))
    plt.plot(loocv_df.binsize,loocv_df.LOOCV,color="black",marker="o")
    plt.xlabel("Bin size (s)")
    plt.ylabel("Logistic Regression \nLOOCV Accuracy")
    plt.tight_layout()
    if save==True:
        plt.savefig(save_path+"lr_loocv_scan.png",dpi=500)
        loocv_df.to_csv(save_path+"lr_loocv.csv")

    #Logistic regression analysis - plot
    binsize=best_bin
    labels_df, n_modules = analysis.label_counter_subgroups(config,0,1200)
    lr, group_labels, label_counts, group_dict, nbins = analysis.lr_labels_timebins(config, labels_df, binsize)
    confusion, class_num, class_labels, accuracy = analysis.loocv_conf_mat(lr, label_counts, group_labels, group_dict)
    plot.plot_conf_mat(confusion, class_num, class_labels,alt_title="Logistic Regression\nConfusion Matrix")
    if save == True:
        plt.savefig(save_path+"lr_confmat.png",dpi=500)

    #BORIS to pose
    BORIS_to_pose_mat, BORIS_to_pose_mat_normalized, loss = analysis.BORIS_to_pose(config)
    plot.BORIS_to_pose_matrix_plot(config, BORIS_to_pose_mat_normalized,figH=2,figW=4)
    if save == True:
        plt.savefig(save_path+"boris_to_pose.png",dpi=500)

    #plt.show()

    #remapping
    labels_df, n_modules = analysis.label_counter_subgroups(config,0,1200)
    BORIS_to_pose_mat, BORIS_to_pose_mat_normalized, loss = analysis.BORIS_to_pose(config)
    labels_df = analysis.make_remappings_from_BORIS(config, labels_df, BORIS_to_pose_mat)
    fig = plot.plot_module_usage_subgroups(config, labels_df, 0, 1200,figW=5,figH=2.5,legend_pos="outside")
    if save==True:
        plt.savefig(save_path+"remapped_labels_subgroups.png",dpi=500)

    #plt.show()

    #plot over time - remapped
    group="k10"
    labels_df, n_modules = analysis.label_counter_subgroups(config,0,60*50,selected_subgroups=[group])
    BORIS_to_pose_mat, BORIS_to_pose_mat_normalized, loss = analysis.BORIS_to_pose(config)
    labels_df = analysis.make_remappings_from_BORIS(config, labels_df, BORIS_to_pose_mat)
    fig = plot.SandPlotClusterFrequency_OverTime(config, labels_df[group],0, 60, 50)
    if save==True:
        plt.savefig(save_path+"remapped_sandplot.png",dpi=500)
