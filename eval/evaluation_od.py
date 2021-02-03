from sklearn.neighbors import LocalOutlierFactor
from pyod.models.iforest import IForest
from pyod.models.hbos import HBOS
from pyod.models.loda import LODA
from pyod.models.copod import COPOD
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
import ast
import eval.evaluation_utils as utils
from sklearn import metrics
from config import eva_root


def evaluation_od_train(x, y, data_name, model_name="iforest", chosen_subspace=None):
    """
    using anomaly detector to yield anomaly score for each subspace,
    generate two files: the subspaces with the highest anomaly score & lof score for each subspace
    :param x: data matrix
    :param y: class information
    :param data_name: the data set name, using for naming the ground truth file
    :param model_name: anomaly detector name, default: lof
    :param chosen_subspace: use this to only evaluate a subset of the power set of full feature space
    :return: df: a ground-truth map using anomaly idx as key and ground truth feature subspace as value.
    """
    global chosen_model

    dim = x.shape[1]
    ano_idx = np.where(y == 1)[0]
    n_ano = len(ano_idx)

    # get all the possible feature subset or just use given subset list
    f_subsets = utils.get_subset_candidate(dim, chosen_subspace)

    # score anomalies in each subspace, generate the score matrix
    n_subsets = len(f_subsets)
    score_matrix = np.zeros([n_ano, n_subsets])
    for i in tqdm(range(n_subsets)):
        subset = f_subsets[i]
        x_subset = x[:, subset]


        if model_name == "iforest":
            clf = IForest()
            clf.fit(x_subset)
            od_score = clf.decision_scores_
        elif model_name == "copod":
            clf = COPOD()
            clf.fit(x_subset)
            od_score = clf.decision_scores_
        elif model_name == "hbos":
            clf = HBOS()
            clf.fit(x_subset)
            od_score = clf.decision_scores_
        else:
            raise ValueError("unsupported od model")

        od_score = utils.min_max_norm(od_score)
        score_matrix[:, i] = od_score[ano_idx]

    if not os.path.exists(eva_root + "data_od_evaluation/"):
        os.makedirs(eva_root + "data_od_evaluation/")

    # score matrix to df
    anomaly_score_df = pd.DataFrame(data=score_matrix, columns=[str(s) for s in f_subsets])
    col_name = anomaly_score_df.columns.tolist()
    col_name.insert(0, 'ano_idx')
    anomaly_score_df["ano_idx"] = ano_idx
    anomaly_score_df = anomaly_score_df.reindex(columns=col_name)
    path1 = eva_root + "data_od_evaluation/" + data_name + "_score_" + model_name + ".csv"
    anomaly_score_df.to_csv(path1, index=False)

    # get the ground truth (one subspace for each anomaly that the anomaly can obtain the highest anomaly score)
    g_truth_df = pd.DataFrame(columns=["ano_idx", "exp_subspace"])

    exp_subspaces = []
    for ii, ano_score in enumerate(score_matrix):
        max_score_idx = int(np.argmax(ano_score))
        exp_subset = str(f_subsets[max_score_idx])
        exp_subspaces.append(exp_subset)
    g_truth_df["ano_idx"] = ano_idx
    g_truth_df["exp_subspace"] = exp_subspaces

    g_truth_df.astype({"exp_subspace": "object"})
    path2 = eva_root + "data_od_evaluation/" + data_name + "_gt_" + model_name + ".csv"
    g_truth_df.to_csv(path2, index=False)
    return anomaly_score_df, g_truth_df


def evaluation_od(exp_subspace_list, x, y, data_name, model_name):
    """
    use outlier detection to evaluate the explanation subspace for each anomaly data object,
    to evaluate whether this subspace is a high-contrast subspace to highlight this anomaly
    i.e., the anomaly detector can or cannot get a higher score in this space
    :param exp_subspace_list: explanation feature subspace for each anomaly, corresponding to ano_idx
    :param x: data set
    :param y: label
    :param data_name: name of dataset
    :param model_name: the name of anomaly detector to generate ground truth
    :return: average precision, jaccard, and anomaly score
    """
    path1 = eva_root + "data_od_evaluation/" + data_name + "_gt_" + model_name + ".csv"
    if not os.path.exists(path1):
        print("annotation file not found, labeling now...")
        _, g_truth_df = evaluation_od_train(x, y, data_name, model_name)
    else:
        g_truth_df = pd.read_csv(path1)

    ano_idx = np.where(y == 1)[0]

    precision_list = np.zeros(len(ano_idx))
    jaccard_list = np.zeros(len(ano_idx))
    recall_list = np.zeros(len(ano_idx))

    for ii, ano in enumerate(ano_idx):
        exp_subspace = list(exp_subspace_list[ii])
        gt_subspace_str = g_truth_df.loc[g_truth_df["ano_idx"] == ano]["exp_subspace"].values[0]
        gt_subspace = ast.literal_eval(gt_subspace_str)

        overlap = list(set(gt_subspace).intersection(set(exp_subspace)))
        union = list(set(gt_subspace).union(set(exp_subspace)))

        precision_list[ii] = len(overlap) / len(exp_subspace)
        jaccard_list[ii] = len(overlap) / len(union)
        recall_list[ii] = len(overlap) / len(gt_subspace)

    return precision_list.mean(), recall_list.mean(), jaccard_list.mean()


def evaluation_od_auc(feature_weight, x, y, data_name, model_name="iforest"):
    """
    use outlier detection to evaluate the explanation subspace for each anomaly data,
    whether this subspace is a high-contrast subspace to highlight this anomaly
    :param exp_subspace_list: explanation feature subspace for each anomaly, corresponding to ano_idx
    :param x: data set
    :param y: label
    :param data_name: name of dataset
    :param model_name: the name of anomaly detector to generate ground truth
    :return: average precision, jaccard, and anomaly score
    """
    path1 = eva_root + "data_od_evaluation/" + data_name + "_gt_" + model_name + ".csv"
    if not os.path.exists(path1):
        print("annotation file not found, labeling now...")
        _, g_truth_df = evaluation_od_train(x, y, data_name, model_name)
    else:
        g_truth_df = pd.read_csv(path1)

    ano_idx = np.where(y == 1)[0]
    dim = x.shape[1]

    auroc_list = np.zeros(len(ano_idx))
    aupr_list = np.zeros(len(ano_idx))
    for ii, ano in enumerate(ano_idx):
        score = feature_weight[ii]

        # ground_truth metrics
        gt_subspace_str = g_truth_df.loc[g_truth_df["ano_idx"] == ano]["exp_subspace"].values[0]
        gt_subspace = ast.literal_eval(gt_subspace_str)
        gt = np.zeros(dim, dtype=int)
        gt[gt_subspace] = 1

        if len(gt_subspace) == dim:
            auroc_list[ii] = 1
            aupr_list[ii] = 1
        else:
            precision, recall, _ = metrics.precision_recall_curve(gt, score)
            aupr_list[ii] = metrics.auc(recall, precision)
            auroc_list[ii] = metrics.roc_auc_score(gt, score)

    return aupr_list.mean(), auroc_list.mean()






