import os
import numpy as np
import pandas as pd
from eval import evaluation_od
from config import root


def main(path):
    data_name = path.split("/")[-1].split(".")[0]
    df = pd.read_csv(path)
    X = df.values[:, :-1]
    y = np.array(df.values[:, -1], dtype=int)
    print("Data name: [%s]" % data_name)

    model_name1 = "hbos"
    path1 = "data_od_evaluation/" + data_name + "_gt_" + model_name1 + ".csv"
    path2 = "data_od_evaluation/" + data_name + "_score_" + model_name1 + ".csv"
    if not (os.path.exists(path1) and os.path.exists(path2)):
        print("OD evaluation model training is processing...")
        evaluation_od.evaluation_od_train(X, y, data_name, model_name1)

    return


if __name__ == '__main__':
    input_root_list = [root + "data/"]
    runs = 1

    for input_root in input_root_list:
        if os.path.isdir(input_root):
            for file_name in sorted(os.listdir(input_root)):
                if file_name.endswith(".csv"):
                    input_path = str(os.path.join(input_root, file_name))
                    name = input_path.split("/")[-1].split('.')[0]
                    main(input_path)

        else:
            input_path = input_root
            name = input_path.split("/")[-1].split(".")[0]
            main(input_path)