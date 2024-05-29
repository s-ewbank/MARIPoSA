#set up paths
import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
mariposa_dir = os.path.dirname(script_dir)
utils_dir = os.path.join(mariposa_dir, 'utils')
sys.path.append(utils_dir)
sys.path.append(mariposa_dir)

#import utils
from utils import analysis, plot, metadata
import matplotlib.pyplot as plt
import importlib

importlib.reload(analysis)
importlib.reload(plot)

config_path="/Users/snewbank/Behavior/PoseVis_test/240319_KPMS-ivket/config.yaml"
save_path="/Users/snewbank/Behavior/PoseVis_test/240319_KPMS-ivket/demo/"


#Load config
config = metadata.load_project(config_path)

#plot without subgroups
labels_df, n_modules = analysis.label_counter_nosubgroups(config,0,1200)
fig = plot.plot_module_usage(config, labels_df, 0, 1200,style="points")
plt.savefig(save_path+"labels_no_subgroups.png",dpi=500)

#plot with subgroups
labels_df, n_modules = analysis.label_counter_subgroups(config,0,1200)
fig = plot.plot_module_usage_subgroups(config, labels_df, 0, 1200)
plt.savefig(save_path+"labels_subgroups.png",dpi=500)

#network comparison plot
comparison_groups=["sal","k10"]
labels_df, n_modules = analysis.label_counter_subgroups(config,0,1200)
fig = plot.network_pairwise_comparison(config, labels_df, 0, 1200, comparison_groups)
plt.savefig(save_path+"network_comparison.png",dpi=500)

#plot over time
group="k10"
labels_df, n_modules = analysis.label_counter_subgroups(config,0,60*50,selected_subgroups=[group])
fig = plot.SandPlotClusterFrequency_OverTime(config, labels_df[group],0, 60, 50)
plt.savefig(save_path+"sandplot.png",dpi=500)

# Linear discriminant analysis
binsize=5*60
labels_df, n_modules = analysis.label_counter_subgroups(config,0,1200)
lda, lda_embeddings, label_counts, group_labels, group_dict, nbins = analysis.lda_labels_timebins(config,labels_df,binsize)
fig = plot.plot_lda(config, lda, lda_embeddings, group_labels, nbins, binsize, cmap="viridis")
plt.savefig(save_path+"lda_embeddings.png",dpi=500)
confusion, class_num, class_labels = analysis.loocv_conf_mat(lda, label_counts, group_labels, group_dict)
plot.plot_conf_mat(confusion, class_num, class_labels)
plt.savefig(save_path+"lda_confmat.png",dpi=500)

#Logistic regression analysis
binsize=5*60
labels_df, n_modules = analysis.label_counter_subgroups(config,0,1200)
lr, group_labels, label_counts, group_dict, nbins = analysis.lr_labels_timebins(config, labels_df, binsize)
confusion, class_num, class_labels = analysis.loocv_conf_mat(lr, label_counts, group_labels, group_dict)
plot.plot_conf_mat(confusion, class_num, class_labels)
plt.savefig(save_path+"lr_confmat.png",dpi=500)

plt.show()

