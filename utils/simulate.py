import numpy as np
from utils import analyze
from scipy.stats import multivariate_normal
from scipy.special import softmax
import pandas as pd

def generate_usage(module_feature_object, n_samples, random_state=42, mode="log-normal", scale=10):
    """
    Generate simulated pose module usage object

    :param module_feature_object: module feature object of class ModuleUsage (from analyze.get_module_usage) or ModuleTransitions (from analyze.get_module_transitions)
    :param n_samples: number of samples to generate
    :param random_state: random state seed (default: 42)
    :param mode: 'log-normal' or 'multivariate_gaussian'; default log-normal
    :param scale: scale factor for variance in log-normal mode (not needed for multivariate_gaussian mode)
    :return: module_feature_object of the same style as the one input

    """
    if mode=="multivariate_gaussian":
        np.random.seed(random_state)

        if module_feature_object.__class__.__name__ == "ModuleUsage":
            X = module_feature_object.label_counts
        elif module_feature_object.__class__.__name__ == "ModuleTransitions":
            X = module_feature_object.transition_counts
        else:
            raise ValueError(
                f'module_feature_object class must be ModuleUsage or ModuleTransitions, not {module_feature_object.__class__}')

        mean_X = np.mean(X, axis=0)
        cov_X = np.cov(X.T)
        simulated_X = np.random.multivariate_normal(mean_X, cov_X, size=n_samples)
        simulated_X[simulated_X < 0] = 0
        simulated_X = simulated_X / np.sum(simulated_X, axis=1)[:, np.newaxis]

        if module_feature_object.__class__.__name__ == "ModuleUsage":
            return analyze.ModuleUsage(simulated_X, ["9999999"] * n_samples, ["9999999"] * n_samples,
                                       module_feature_object.feat_names, module_feature_object.group_dict, None)
        elif module_feature_object.__class__.__name__ == "ModuleTransitions":
            print("Warning - generating module transitions currently not tested for this function")
            return analyze.ModuleTransitions(simulated_X, simulated_X, ["9999999"] * n_samples, ["9999999"] * n_samples,
                                             module_feature_object.feat_names, module_feature_object.group_dict, None)

    if mode == "log-normal":
        np.random.seed(random_state)

        if module_feature_object.__class__.__name__ == "ModuleUsage":
            X = module_feature_object.label_counts
        elif module_feature_object.__class__.__name__ == "ModuleTransitions":
            X = module_feature_object.transition_counts
        else:
            raise ValueError(
                f'module_feature_object class must be ModuleUsage or ModuleTransitions, not {module_feature_object.__class__}')

        mean_X = np.mean(X, axis=0)
        cov_X = np.cov(X.T) * (scale**2)
        Z = np.random.multivariate_normal(mean=mean_X, cov=cov_X, size=n_samples)
        simulated_X = softmax(Z, axis=1)

        if module_feature_object.__class__.__name__ == "ModuleUsage":
            return analyze.ModuleUsage(simulated_X, ["9999999"] * n_samples, ["9999999"] * n_samples,
                                       module_feature_object.feat_names, module_feature_object.group_dict, None)
        elif module_feature_object.__class__.__name__ == "ModuleTransitions":
            print("Warning - generating module transitions currently not tested for this function")
            return analyze.ModuleTransitions(simulated_X, simulated_X, ["9999999"] * n_samples, ["9999999"] * n_samples,
                                             module_feature_object.feat_names, module_feature_object.group_dict, None)

def generate_usage_labeled(module_feature_object, n_samples_per_bin, bins, regression, max_iters="default",
                           random_state=42, mode="log-normal", scale=10, verbosity="medium"):
    """
    Generate simulated pose module usage object

    :param module_feature_object: module feature object of class ModuleUsage (from analyze.get_module_usage) or ModuleTransitions (from analyze.get_module_transitions)
    :param n_samples_per_bin: number of samples to generate per bin in timebin
    :param bins: 2D array of bins containing upper and lower limit for each bin
    :param regression: regression model to label samples
    :param max_iters: number for how many iterations to attempt to generate samples; will raise an error if exceeded without generating enough samples; number or 'default' for n_samples_total*10
    :param random_state: random state seed (default: 42)
    :param mode: 'log-normal' or 'multivariate_gaussian'; default log-normal
    :param scale: scaling factor for covariance in log-transformed approach
    :param verbosity: 'low','medium', or 'high'
    :return: module_feature_object of the same style as the one input

    """
    if max_iters=="default":
        max_iters = bins.shape[0] * n_samples_per_bin * 10

    if max_iters<250:
        print_interval = 5
    elif max_iters<2000:
        print_interval = 50
    else:
        print_interval = 500

    if mode=="multivariate_gaussian":
        np.random.seed(random_state)

        if module_feature_object.__class__.__name__ == "ModuleUsage":
            X = module_feature_object.label_counts
        elif module_feature_object.__class__.__name__ == "ModuleTransitions":
            X = module_feature_object.transition_counts
        else:
            raise ValueError(
                f'module_feature_object class must be ModuleUsage or ModuleTransitions, not {module_feature_object.__class__}')

        mean_X = np.mean(X, axis=0)
        cov_X = np.cov(X.T)
        simulated_X = []
        group_labels=[]
        n_in_bins = np.zeros(bins.shape[0])
        count=0
        while np.sum(n_in_bins < n_samples_per_bin)>0:
            count+=1
            if verbosity=="high":
                if count%print_interval==0:
                    print(f"Iter {count}: {n_in_bins}")
            if verbosity=="medium":
                if count%print_interval==0:
                    perc_samples_gen = 100 * (np.sum(n_in_bins)/(n_samples_per_bin * bins.shape[0]))
                    print(f"Iter {count}: {perc_samples_gen:.1f}% samples generated ({np.sum(n_in_bins < n_samples_per_bin)} of {bins.shape[0]} bins not filled)")
            simulated_X_i = np.random.multivariate_normal(mean_X, cov_X, size=1)
            simulated_X_i[simulated_X_i < 0] = 0
            simulated_X_i = simulated_X_i / np.sum(simulated_X_i, axis=1)[:, np.newaxis]
            for b in range(bins.shape[0]):
                if n_in_bins[b] < n_samples_per_bin:
                    if ((regression.predict(simulated_X_i)>bins[b,0]) and (regression.predict(simulated_X_i)<bins[b,1])):
                        simulated_X.append(simulated_X_i)
                        n_in_bins[b]=n_in_bins[b]+1
                        group_labels.append(bins[b,0])
            if count>max_iters:
                raise ValueError(f"Maximum number of iterations ({max_iters}) reached!")

        simulated_X = np.array(simulated_X)
        simulated_X = np.squeeze(simulated_X,axis=1)

        if module_feature_object.__class__.__name__ == "ModuleUsage":
            return analyze.ModuleUsage(simulated_X, group_labels, group_labels,
                                       module_feature_object.feat_names, {i:i for i in bins[:,0]}, None)
        elif module_feature_object.__class__.__name__ == "ModuleTransitions":
            print("Warning - generating module transitions currently not tested for this function")
            return analyze.ModuleTransitions(simulated_X, np.nan, group_labels,
                                             module_feature_object.feat_names, {i:i for i in bins[:,0]}, None)

    if mode == "log-normal":
        np.random.seed(random_state)

        if module_feature_object.__class__.__name__ == "ModuleUsage":
            X = module_feature_object.label_counts
        elif module_feature_object.__class__.__name__ == "ModuleTransitions":
            X = module_feature_object.transition_counts
        else:
            raise ValueError(
                f'module_feature_object class must be ModuleUsage or ModuleTransitions, not {module_feature_object.__class__}')

        mean_X = np.mean(X, axis=0)
        cov_X = np.cov(X.T) * (scale**2)
        simulated_X = []
        group_labels=[]
        n_in_bins = np.zeros(bins.shape[0])
        count=0
        while np.sum(n_in_bins < n_samples_per_bin)>0:
            count+=1
            if verbosity=="high":
                if count%print_interval==0:
                    print(f"Iter {count}: {n_in_bins}")
            if verbosity=="medium":
                if count%print_interval==0:
                    perc_samples_gen = 100 * (np.sum(n_in_bins)/(n_samples_per_bin * bins.shape[0]))
                    print(f"Iter {count}: {perc_samples_gen:.1f}% samples generated ({np.sum(n_in_bins < n_samples_per_bin)} of {bins.shape[0]} bins not filled)")
            Z = np.random.multivariate_normal(mean=mean_X, cov=cov_X, size=1)
            simulated_X_i = softmax(Z, axis=1)
            for b in range(bins.shape[0]):
                if n_in_bins[b] < n_samples_per_bin:
                    if ((regression.predict(simulated_X_i)>bins[b,0]) and (regression.predict(simulated_X_i)<bins[b,1])):
                        simulated_X.append(simulated_X_i)
                        n_in_bins[b]=n_in_bins[b]+1
                        group_labels.append(bins[b,0])
            if count>max_iters:
                raise ValueError(f"Maximum number of iterations ({max_iters}) reached!")

        else:
            return ValueError("Mode invalid! Must be log-normal or multivariate_gaussian")

        simulated_X = np.array(simulated_X)
        simulated_X = np.squeeze(simulated_X,axis=1)

        if module_feature_object.__class__.__name__ == "ModuleUsage":
            return analyze.ModuleUsage(simulated_X, group_labels, group_labels,
                                       module_feature_object.feat_names, {i:i for i in bins[:,0]}, None)
        elif module_feature_object.__class__.__name__ == "ModuleTransitions":
            print("Warning - generating module transitions currently not tested for this function")
            return analyze.ModuleTransitions(simulated_X, np.nan, group_labels,
                                             module_feature_object.feat_names, {i:i for i in bins[:,0]}, None)

def generate_sequence(config, labels_df, T, n_subjs = 1, random_state=42, verbose=False):
    """
    Generate individual simulated pose module label sequence of length T (noting that T = number of observations, not time)

    :param config: the config object
    :param labels_df: labels_df from analyze.get_module_labels
    :param T: length of sequence to be generated
    :param n_subjs: number of subjects to generate (if labels_df has subgroups, will be number of subjects PER GROUP)
    :param random_state: random state seed (default: 42)
    :param verbose: print progress (default: False)
    :return: sequence

    """
    if verbose:
        print("Analyzing input labels dataframe")
    np.random.seed(random_state)
    module_usage = analyze.get_module_usage(config, labels_df)
    module_transitions = analyze.get_module_transitions(config, labels_df)
    subgroups = np.unique(module_usage.group_labels)
    n_subgroups = len(subgroups)
    if n_subgroups==1:
        start_prob = np.mean(module_usage.label_counts,axis=0)
        transition_mat = np.mean(module_transitions.transition_count_matrices,axis=0)
        transition_mat = transition_mat / np.sum(transition_mat, axis=1, keepdims=True)
        row_sums = np.sum(transition_mat, axis=1)
        transition_mat = transition_mat / row_sums[:, np.newaxis]
        transition_mat[np.abs(np.sum(transition_mat, axis=1) - 1) > 1e-6] = 1
        header = []
        for i in range(n_subjs):
            if verbose:
                print(f"Generating data for subject {i+1} of {n_subjs}")
            current_state = np.random.choice(len(start_prob), p=start_prob)
            sequence = [current_state]
            for _ in range(1, T):
                current_state = np.random.choice(len(transition_mat), p=transition_mat[current_state])
                sequence.append(current_state)
            subj_i = f"subj_{i}"
            header.append(subj_i)
            labels_i = np.array(sequence)
            if i == 0:
                sim_labels_df = pd.DataFrame(labels_i, columns=[header])
            else:
                sim_labels_df.insert(loc=i, value=labels_i, column=subj_i)
                sim_labels_df = sim_labels_df.copy()
    else:
        count=0
        total_subjs = int(n_subjs) * int(n_subgroups)
        reverse_group_dict = {v: u for u, v in module_usage.group_dict.items()}
        header1=[]
        header2=[]
        for g in range(n_subgroups):
            start_prob = np.mean(module_usage.label_counts[np.array(module_usage.group_labels)==g],axis=0)
            sub_transition_count_matrices = np.array(module_transitions.transition_count_matrices)[np.array(module_transitions.group_labels)==g,:,:]
            transition_mat = np.mean(sub_transition_count_matrices,axis=0)
            transition_mat = transition_mat / np.sum(transition_mat, axis=1, keepdims=True)
            row_sums = np.sum(transition_mat, axis=1)
            transition_mat = transition_mat / row_sums[:, np.newaxis]
            transition_mat[np.abs(np.sum(transition_mat, axis=1) - 1) > 1e-6] = 1
            group = f"{reverse_group_dict[g]}"
            for i in range(n_subjs):
                subj_i = f"{group}_{i}"
                if verbose:
                    print(f"Generating data for {subj_i} ({count+1} of {total_subjs})")
                header = subj_i
                header2.append(header)
                header1.append(group)
                current_state = np.random.choice(len(start_prob), p=start_prob)
                sequence = [current_state]
                for _ in range(1, T):
                    current_state = np.random.choice(len(transition_mat), p=transition_mat[current_state])
                    sequence.append(current_state)
                labels_i = np.array(sequence)
                if count == 0:
                    sim_labels_df = pd.DataFrame(labels_i, columns=[header])
                else:
                    sim_labels_df.insert(loc=count, value=labels_i, column=header)
                    sim_labels_df = sim_labels_df.copy()
                count+=1
        sim_labels_df.columns = [header1, header2]
    return sim_labels_df