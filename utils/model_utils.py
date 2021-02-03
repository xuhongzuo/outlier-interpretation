import numpy as np
import math


def weight2subspace(weight, ratio=0.7, num=-1):
    """
    this function is to transfer feature weight list to a feature subspace with higher weight
    given different ratio (ratio of weight summation of subspace to the full space) of subspace length
    :param weight:
    :param ratio:
    :param num:
    :return:
    """
    dim = len(weight)

    threshold = ratio * np.sum(weight)

    sorted_idx = np.argsort(weight)
    sorted_idx = [sorted_idx[dim - i - 1] for i in range(dim)]

    if num != -1:
        exp_subspace = sorted_idx[:num]
        exp_subspace = list(np.sort(exp_subspace))
        return exp_subspace

    tmp_s = 0
    exp_subspace = []
    for idx in sorted_idx:
        tmp_s += weight[idx]
        exp_subspace.append(idx)
        if tmp_s >= threshold:
            break
    exp_subspace = list(np.sort(exp_subspace))
    return exp_subspace


def weight2subspace_pn(weight):
    exp_subspace = []
    for i in range(len(weight)):
        if weight[i] > 0:
            exp_subspace.append(i)
    if len(exp_subspace) == 0:
        exp_subspace.append(np.argsort(weight)[len(weight) - 1])
    exp_subspace = list(np.sort(exp_subspace))
    return exp_subspace


def get_exp_subspace(fea_weight_lst, w2s_ratio, real_exp_len=None):
    exp_subspace_lst = []
    n_ano = len(fea_weight_lst)
    dim = len(fea_weight_lst[0])

    for ii in range(n_ano):
        fea_weight = fea_weight_lst[ii]
        if w2s_ratio == "real_len":
            if real_exp_len is None:
                raise ValueError("not give real exp len")
            exp_subspace_lst.append(weight2subspace(fea_weight, num=real_exp_len[ii]))

        elif w2s_ratio == "auto":
            r = math.sqrt(2 / dim)
            exp_subspace_lst.append(weight2subspace(fea_weight, ratio=r))

        elif w2s_ratio == "pn":
            exp_subspace_lst.append(weight2subspace_pn(fea_weight))

        else:
            exp_subspace_lst.append(weight2subspace(fea_weight, ratio=w2s_ratio))
    return exp_subspace_lst


