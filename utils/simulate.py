import numpy as np
from utils import analyze
from scipy.special import softmax

def generate_usage(module_feature_object, n_samples):
    if module_feature_object.__class__.__name__=="ModuleUsage":
        X=module_feature_object.label_counts
    elif module_feature_object.__class__.__name__=="ModuleTransitions":
        X=module_feature_object.transition_counts
    else:
        raise ValueError(f'module_feature_object class must be ModuleUsage or ModuleTransitions, not {module_feature_object.__class__}')
    mean_X = np.mean(X, axis=0)
    cov_X = np.cov(X.T)
    simulated_X = np.random.multivariate_normal(mean_X, cov_X, size=n_samples)
    simulated_X[simulated_X < 0] = 0
    simulated_X = simulated_X / np.sum(simulated_X, axis=1)[:,np.newaxis]
    return analyze.ModuleUsage(simulated_X, ["9999999"]*n_samples, ["9999999"]*n_samples, module_feature_object.feat_names, module_feature_object.group_dict, None)

def generate_sequence(config, labels_df, T):
    module_usage = analyze.get_module_usage(config, labels_df)
    module_transitions = analyze.get_module_transitions(config, labels_df)
    start_prob = np.mean(module_usage.label_counts,axis=0)
    transition_mat = np.mean(module_transitions.transition_count_matrices,axis=0)
    transition_mat = transition_mat / np.sum(transition_mat, axis=1, keepdims=True)
    row_sums = np.sum(transition_mat, axis=1)
    transition_mat = transition_mat / row_sums[:, np.newaxis]
    transition_mat[np.abs(np.sum(transition_mat, axis=1) - 1) > 1e-6] = 1
    current_state = np.random.choice(len(start_prob), p=start_prob)
    sequence = [current_state]
    for _ in range(1, T):
        current_state = np.random.choice(len(transition_mat), p=transition_mat[current_state])
        sequence.append(current_state)
    return sequence