import numpy as np


def print_eval_runs2(runs_metric_lst, data_name, algo_name):
    runs_metric_lst = np.array(runs_metric_lst)
    precision, recall, jaccard, t = np.average(runs_metric_lst, axis=0)
    txt = "%s, od_eval, [p,r,j], %.4f, %.4f, %.4f, time, %.2f, %s" % \
          (data_name, precision, recall, jaccard, t, algo_name)
    return txt


def print_eval_runs(runs_metric_lst, data_name, algo_name):
    runs_metric_lst = np.array(runs_metric_lst)
    precision, recall, jaccard, aupr, auroc, t = np.average(runs_metric_lst, axis=0)
    txt = "%s, [p r j aupr auroc], %.4f, %.4f, %.4f, %.4f, %.4f, time, %.2f, %s" % \
          (data_name, precision, recall, jaccard, aupr, auroc, t, algo_name)
    return txt

