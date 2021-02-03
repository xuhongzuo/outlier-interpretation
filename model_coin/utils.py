import os
import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest


def detect_lof(args, X):
    num_inst = X.shape[0]
    num_nbr = int(num_inst * args.ratio_nbr)
    clf = LocalOutlierFactor(n_neighbors=num_nbr)
    y_pred = clf.fit_predict(X)
    outlier_scores = -clf.negative_outlier_factor_

    return y_pred


def detect_isoforest(args, X):
    num_inst = X.shape[0]
    clf = IsolationForest(behaviour='new', max_samples=num_inst, random_state=0)
    clf.fit(X)
    y_pred = clf.predict(X)
    outlier_scores = -clf.decision_function(X)

    return y_pred


def get_datast_basic_info(path):
    data_name = path.split("/")[-1].split(".")[0]
    df = pd.read_csv(path)
    x = df.values[:, :-1]
    y = np.array(df.values[:, -1], dtype=int)
    n = x.shape[0]
    dim = x.shape[1]
    n_ano = len(np.where(y == 1)[0])
    ratio_ano = n_ano / n

    print("%s, %d, %d, %d, %.4f " % (data_name, n, dim, n_ano, ratio_ano))

    return


if __name__ == '__main__':
    input_root_list = ["E:/OneDrive/work/0data/odds/integer/"]

    seed = -1

    for input_root in input_root_list:
        if os.path.isdir(input_root):
            for file_name in sorted(os.listdir(input_root)):
                if file_name.endswith(".csv"):
                    input_path = str(os.path.join(input_root, file_name))
                    name = input_path.split("/")[-1].split('.')[0]
                    get_datast_basic_info(input_path)

        else:
            input_path = input_root
            name = input_path.split("/")[-1].split(".")[0]
            get_datast_basic_info(input_path)