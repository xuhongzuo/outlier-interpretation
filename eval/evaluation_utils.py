import numpy as np


def powerset(s):
    x = len(s)
    pw_set = []
    for i in range(1 << x):
        pw_set.append([s[j] for j in range(x) if (i & (1 << j))])
    return pw_set


def min_max_norm(array):
    if np.min(array) == np.max(array):
        return array * 0
    else:
        return (array - np.min(array))/(np.max(array) - np.min(array))


def get_subset_candidate(dim, chosen_subspace=None):
    if chosen_subspace is not None:
        f_subsets = []
        for subset in chosen_subspace:
            subset = list(subset)
            if subset not in f_subsets:
                f_subsets.append(list(subset))
    else:
        full_set = np.arange(dim)
        f_subsets = powerset(full_set)
        f_subsets.remove([])
    return f_subsets
