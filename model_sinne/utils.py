import numpy as np


def powerset(s):
    x = len(s)
    pw_set = []
    for i in range(1 << x):
        pw_set.append([s[j] for j in range(x) if (i & (1 << j))])
    return pw_set

