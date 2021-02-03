import os
import ast
import time, datetime
import argparse
import pandas as pd
import numpy as np
from prettytable import PrettyTable
from model_sinne.SiNNE import SiNNE
from model_iml.Anchor import Anchor
from config import root
from eval.evaluation_od import evaluation_od
from utils.eval_print_utils import print_eval_runs2


# ------------------- parameters ----------------- #
algorithm_name = "anchor"

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default="data/")
parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--eval', type=ast.literal_eval, default=True, help='')
if algorithm_name == "sinne":
    parser.add_argument('--max_level', default='full', help='')
    parser.add_argument("--width", type=int, default=10, help='')
    parser.add_argument("--ensemble_num", type=int, default=100, help='')
    parser.add_argument("--sample_num", type=int, default=8, help='')
    parser.add_argument("--pretrain", type=bool, default=False, help='')
    parser.add_argument("--verbose", type=bool, default=False, help='')
elif algorithm_name == 'anchor':
    parser.add_argument('--kernel', default='rbf', help='')
else:
    raise NotImplementedError("not supported algorithm")
args = parser.parse_args()

input_root_list = [root + args.path]
od_eval_model = ["iforest", "copod", "hbos"]
runs = args.runs
record_name = ""

# ------------------- record ----------------- #
if not os.path.exists("record/" + algorithm_name):
    os.makedirs("record/" + algorithm_name)
record_path = "record/" + algorithm_name + "/zout." + \
              algorithm_name + "." + record_name + ".txt"
doc = open(record_path, 'a')
tab1 = PrettyTable(["parameter", "value"])
tab1.add_row(["@ data", str(input_root_list)])
tab1.add_row(["@ algorithm_name", str(algorithm_name)])
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
    data_name = path.split("/")[-1].split(".")[0]

    # this is to remove the prefix index number of data set name, so that we can match the annotation file.
    data_name = data_name[3:]

    print("# ------------------ %s ------------------ # " % data_name)

    df = pd.read_csv(path)
    X = df.values[:, :-1]
    y = np.array(df.values[:, -1], dtype=int)

    runs_metric_lst = [[] for k in range(len(od_eval_model))]
    for i in range(run_times):
        print("runs: %d" % (i + 1))
        time1 = time.time()

        if algorithm_name == "sinne":
            model = SiNNE(max_level=args.max_level, width=args.width, ensemble_num=args.ensemble_num,
                          sample_num=args.sample_num, pretrain=args.pretrain)
            exp_subspace_list = model.fit(X, y)
        elif algorithm_name == 'anchor':
            model = Anchor()
            exp_subspace_list = model.fit(X, y)
        else:
            raise NotImplementedError("not implemented the algorithm")
        t = time.time() - time1

        if args.eval:
            # ---------------------- evaluation -------------------------- #
            for mm, eval_model in enumerate(od_eval_model):
                precision, recall, jaccard = evaluation_od(exp_subspace_list, X, y, data_name, model_name=eval_model)
                metric_lst = [precision, recall, jaccard, t]
                runs_metric_lst[mm].append(metric_lst)
                print("{}, eval_model: {}, {}".format(data_name, eval_model, metric_lst))

    if args.eval:
        for mm in range(len(od_eval_model)):
            name = path.split("/")[-1].split(".")[0]
            txt = print_eval_runs2(runs_metric_lst[mm], data_name=name, algo_name=algorithm_name)
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
