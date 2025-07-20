import numpy as np
from utils import analyze

def generate_usage(module_feature_object, n_samples, n_bootstraps=10, bootstrap_fraction=0.3, random_state=42):
    """
    Generate simulated pose module usage object

    :param module_feature_object: module feature object of class ModuleUsage (from analyze.get_module_usage) or ModuleTransitions (from analyze.get_module_transitions)
    :param n_samples: number of samples to generate
    :param random_state: random state seed (default: 42
    :return: module_feature_object of the same style as the one input

    """
    np.random.seed(random_state)

    if module_feature_object.__class__.__name__ == "ModuleUsage":
        X = module_feature_object.label_counts
    elif module_feature_object.__class__.__name__ == "ModuleTransitions":
        X = module_feature_object.transition_counts
    else:
        raise ValueError(
            f'module_feature_object class must be ModuleUsage or ModuleTransitions, not {module_feature_object.__class__}')

    simulated_X = []
    for boot in range(n_bootstraps):
        boot = np.random.randint(0, high=X.shape[0], size=int(X.shape[0] * bootstrap_fraction))
        X_b = X[boot, :]
        mean_X = np.mean(X_b, axis=0)
        cov_X = np.cov(X_b.T)
        simulated_X_b = np.random.multivariate_normal(mean_X, cov_X, size=int(n_samples / n_bootstraps))
        simulated_X_b[simulated_X_b < 0] = 0
        simulated_X_b = simulated_X_b / np.sum(simulated_X_b, axis=1)[:, np.newaxis]
        simulated_X.append(simulated_X_b)

    simulated_X = np.concatenate(simulated_X, axis=0)

    if module_feature_object.__class__.__name__ == "ModuleUsage":
        return analyze.ModuleUsage(simulated_X, ["9999999"] * n_samples, ["9999999"] * n_samples,
                                   module_feature_object.feat_names, module_feature_object.group_dict, None)
    elif module_feature_object.__class__.__name__ == "ModuleTransitions":
        print("Warning - generating module transitions currently not tested for this function")
        return analyze.ModuleTransitions(simulated_X, simulated_X, ["9999999"] * n_samples, ["9999999"] * n_samples,
                                         module_feature_object.feat_names, module_feature_object.group_dict, None)


def generate_sequence(config, labels_df, T, random_state=42):
    """
    Generate individual simulated pose module label sequence of length T (noting that T = number of observations, not time)

    :param config: the config object
    :param labels_df: labels_df from analyze.get_module_labels
    :param T: length of sequence to be generated
    :param random_state: random state seed (default: 42
    :return: sequence

    """
    np.random.seed(random_state)
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