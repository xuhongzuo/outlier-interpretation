import sklearn
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from alibi.explainers import AnchorTabular
from tqdm import tqdm


class Anchor:
    def __init__(self, kernel="rbf"):
        """

        :param kernel: clf model svm parameter
        :param threshold: threshold is used to filter feature subset for each data, the shap values of selected feature
        subspace accounts for [threshold] of the sum of the shap values of feature full space.
        """
        self.ano_idx = None

        self.kernel = kernel

        self.dim = None
        return

    def fit(self, x, y):

        self.dim = x.shape[1]

        # clf = sklearn.svm.SVC(kernel=self.kernel, probability=True)
        clf = RandomForestClassifier()
        clf.fit(x, y)

        y_pred = clf.predict(x)
        print("Clf model accuracy: [{:.4f}]".format(sklearn.metrics.accuracy_score(y, y_pred)))

        self.ano_idx = np.where(y == 1)[0]
        print(self.ano_idx.shape)

        n_f = x.shape[1]
        feature_names = ["A"+str(i) for i in range(n_f)]
        # use anchor
        predict_fn = lambda xx: clf.predict_proba(xx)
        explainer = AnchorTabular(predict_fn, feature_names)
        explainer.fit(x, disc_perc=(25, 50, 75))

        exp_sub_lst = []
        for i in tqdm(range(len(self.ano_idx))):
            ano = x[self.ano_idx[i]]
            explanation = explainer.explain(ano, threshold=0.95)
            anchor = explanation['anchor']
            f_sub = []
            for a in anchor:
                for item in a.split(" "):
                    if item.startswith("A"):
                        item = int(item[1:])
                        f_sub.append(item)
            # print(anchor, f_sub)
            if len(f_sub) == 0:
                f_sub = np.arange(n_f)
            exp_sub_lst.append(f_sub)

        return exp_sub_lst


import pandas as pd
path = "../data/00-pima.csv"
df = pd.read_csv(path)
X = df.values[:, :-1]
y = np.array(df.values[:, -1], dtype=int)
model = Anchor()
exp_sub_lst = model.fit(X, y)
print(len(exp_sub_lst))