from utils import analysis, plot, metadata
import matplotlib.pyplot as plt
import importlib

importlib.reload(analysis)
importlib.reload(plot)

config_path="/Users/snewbank/Behavior/PoseVis_test/240216_test/config.yaml"

#Load config
config = metadata.load_project(config_path)

#plot without subgroups
labels_df, n_modules = analysis.label_counter_nosubgroups(config,0,1200)
plot.plot_module_usage(config, labels_df, 0, 1200,style="points")

#plot with subgroups
labels_df, n_modules = analysis.label_counter_subgroups(config,0,1200)
plot.plot_module_usage_subgroups(config, labels_df, 0, 1200)

#network comparison plot
comparison_groups=["sal","k10"]
labels_df, n_modules = analysis.label_counter_subgroups(config,0,1200)
plot.network_pairwise_comparison(config, labels_df, 0, 1200, comparison_groups)

#plot over time
group="k10"
labels_df, n_modules = analysis.label_counter_subgroups(config,0,60*50,selected_subgroups=[group])
plot.SandPlotClusterFrequency_OverTime(config, labels_df[group],0, 60, 50)

# Linear discriminant analysis
binsize=5*60
labels_df, n_modules = analysis.label_counter_subgroups(config,0,1200)
lda, lda_embeddings, group_labels, nbins = analysis.lda_labels_timebins(config,labels_df,binsize)
plot.plot_lda(config, lda, lda_embeddings, group_labels, nbins, binsize, cmap="viridis")
plt.show()

