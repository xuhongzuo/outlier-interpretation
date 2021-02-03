import os
import ast
import glob
import time, datetime
import argparse
import pandas as pd
import numpy as np
from prettytable import PrettyTable

from model_aton.ATON import ATON
from model_aton.ATON_ablation import ATONabla
from model_aton.ATON_ablation2 import ATONabla2
from model_aton.ATON_ablation3 import ATONabla3
from model_iml.SHAP import SHAP
from model_iml.LIME import LIME
from model_coin.COIN import COIN
from model_iml.IntGrad import IntGrad

from utils import model_utils
from utils.eval_print_utils import print_eval_runs
from eval.evaluation_od import evaluation_od, evaluation_od_auc
from config import root, eva_root, get_parser
import warnings
warnings.filterwarnings("ignore")


# ------------------- parser ----------------- #
# this script can perform outlier interpretation method ATON, COIN, SHAP, and LIME, IntGrad
algorithm_name = "intgrad"
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=ast.literal_eval, default=True)
parser.add_argument('--eval', type=ast.literal_eval, default=True)
parser.add_argument('--path', type=str, default="data/")
parser.add_argument('--w2s_ratio', type=str, default='real_len')
parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--record_name', type=str, default='')
parser = get_parser(algorithm_name, parser)
args = parser.parse_args()


input_root_list = [root + args.path]
w2s_ratio = args.w2s_ratio
od_eval_model = ["iforest", "copod", "hbos"]
runs = args.runs
record_name = args.record_name


# ------------------- record ----------------- #
if not os.path.exists("record/" + algorithm_name):
    os.makedirs("record/" + algorithm_name)
if not os.path.exists("checkpoints"):
    os.makedirs("checkpoints/")
record_path = "record/" + algorithm_name + "/zout." + \
              algorithm_name + "." + record_name + ".txt"
doc = open(record_path, 'a')
tab1 = PrettyTable(["parameter", "value"])
tab1.add_row(["@ data", str(input_root_list)])
tab1.add_row(["@ algorithm_name", str(algorithm_name)])
tab1.add_row(["@ w2s_ratio", str(w2s_ratio)])
tab1.add_row(["@ runs", str(runs)])
tab1.add_row(["@ od_eval_model", str(od_eval_model)])
tab1.add_row(["@ start_time", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
for k in list(vars(args).keys()):
    tab1.add_row([k, vars(args)[k]])
print(tab1, file=doc)
print(tab1)
doc.close()
time.sleep(0.2)


def main(path, run_times):
    print("eval:", args.eval)
    print("gpu :", args.gpu)
    data_name = path.split("/")[-1].split(".")[0]

    # this is to remove the prefix index number of data set name, so that we can match the annotation file.
    data_name = data_name[3:]

    print("# ------------------ %s ------------------ # " % data_name)

    df = pd.read_csv(path)
    X = df.values[:, :-1]
    y = np.array(df.values[:, -1], dtype=int)

    real_len_lst = []
    runs_metric_lst = [[] for k in range(len(od_eval_model))]
    if args.eval and args.w2s_ratio == "real_len":
        gt_lst = []
        for eval_m in od_eval_model:
            folder = eva_root + "data_od_evaluation/"
            gt_path = os.path.join(folder, data_name + "_gt_" + eval_m + ".csv")
            if len(glob.glob(gt_path)) == 0:
                raise FileNotFoundError("no such gt file:" + gt_path)
            gt_str = pd.read_csv(gt_path)["exp_subspace"].values
            gt_lst.append([ast.literal_eval(gtt) for gtt in gt_str])

        for gt in gt_lst:
            real_len_lst.append([len(gtt) for gtt in gt])

    t = 0
    for i in range(run_times):
        print("runs: %d" % (i + 1))
        time1 = time.time()

        # ------------ run the chosen algorithm to get feature weight ------------- #
        fea_weight_lst = run_model(algorithm_name, X, y)

        # ------------------- transfer feature weight to subspace ----------------- #
        subspace_outputs = []
        if args.eval:
            for j in range(len(od_eval_model)):
                if w2s_ratio == "real_len":
                    real_len = real_len_lst[j]
                    subspace = model_utils.get_exp_subspace(fea_weight_lst, w2s_ratio=w2s_ratio, real_exp_len=real_len)
                else:
                    subspace = model_utils.get_exp_subspace(fea_weight_lst, w2s_ratio=w2s_ratio)
                subspace_outputs.append(subspace)

        t = time.time() - time1

        # ---------------------- evaluation -------------------------- #
        if args.eval:
            for mm, eval_model in enumerate(od_eval_model):
                p, j, s = evaluation_od(subspace_outputs[mm], X, y, data_name, eval_model)
                auroc, aupr = evaluation_od_auc(fea_weight_lst, X, y, data_name, model_name=eval_model)
                metric_lst = [p, j, s, auroc, aupr, t]
                runs_metric_lst[mm].append(metric_lst)
                print("data: {}, eval_model: {}, {}".format(path.split("/")[-1].split(".")[0], eval_model, metric_lst))

    if args.eval:
        name = path.split("/")[-1].split(".")[0]
        for mm in range(len(od_eval_model)):
            txt = print_eval_runs(runs_metric_lst[mm], data_name=name, algo_name=algorithm_name)
            print(txt)

            doc = open(record_path, 'a')
            print(txt, file=doc)
            doc.close()
    else:
        txt = data_name + "," + str(round(t, 2)) + "," + algorithm_name
        print(txt)
        doc = open(record_path, 'a')
        print(txt, file=doc)
        doc.close()
    return


def run_model(algorithm, X, y):
    if algorithm == "aton":
        model = ATON(verbose=False, gpu=args.gpu,
                     nbrs_num=args.nbrs_num, rand_num=args.rand_num,
                     alpha1=args.alpha1, alpha2=args.alpha2,
                     n_epoch=args.n_epoch, batch_size=args.batch_size, lr=args.lr,
                     n_linear=args.n_linear, margin=args.margin)
        fea_weight_lst = model.fit(X, y)

    elif algorithm == "aton_ablation":
        model = ATONabla(verbose=False,
                         nbrs_num=args.nbrs_num, rand_num=args.rand_num, n_epoch=args.n_epoch,
                         batch_size=args.batch_size, lr=args.lr, n_linear=args.n_linear, margin=args.margin)
        fea_weight_lst = model.fit(X, y)

    elif algorithm == "aton_ablation2":
        model = ATONabla2(verbose=False,
                          nbrs_num=args.nbrs_num, rand_num=args.rand_num, n_epoch=args.n_epoch,
                          batch_size=args.batch_size, lr=args.lr, margin=args.margin)
        fea_weight_lst = model.fit(X, y)

    elif algorithm == "aton_ablation3":
        model = ATONabla3(verbose=False, gpu=True,
                          nbrs_num=args.nbrs_num, rand_num=args.rand_num, n_epoch=args.n_epoch,
                          batch_size=args.batch_size, lr=args.lr, n_linear=args.n_linear, margin=args.margin)
        fea_weight_lst = model.fit(X, y)

    elif algorithm == "shap":
        model = SHAP(kernel=args.kernel, n_sample=args.n_sample, threshold=args.threshold)
        fea_weight_lst = model.fit(X, y)

    elif algorithm == "lime":
        model = LIME(discretize_continuous=args.discretize_continuous, discretizer=args.discretizer)
        fea_weight_lst = model.fit(X, y)

    elif algorithm == "intgrad":
        model = IntGrad(n_steps=args.n_steps, method=args.method)
        fea_weight_lst = model.fit(X, y)

    elif algorithm == "coin":
        sgnf_prior = 1
        model = COIN(X, y, args.ratio_nbr, AUG=args.AUG, MIN_CLUSTER_SIZE=args.MIN_CLUSTER_SIZE,
                     MAX_NUM_CLUSTER=args.MAX_NUM_CLUSTER, VAL_TIMES=args.VAL_TIMES,
                     C_SVM=args.C_SVM, THRE_PS=args.THRE_PS, DEFK=args.DEFK)
        fea_weight_lst = model.fit(sgnf_prior)
    else:
        raise NotImplementedError("not implemented the algorithm")
    return fea_weight_lst


if __name__ == '__main__':
    for input_root in input_root_list:
        if os.path.isdir(input_root):
            for file_name in sorted(os.listdir(input_root)):
                if file_name.endswith(".csv"):
                    input_path = str(os.path.join(input_root, file_name))
                    main(input_path, runs)

        else:
            input_path = input_root
            main(input_path, runs)
