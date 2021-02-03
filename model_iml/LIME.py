import lime
import lime.lime_tabular
import numpy as np
import sklearn
import math
from tqdm import tqdm
import sklearn.datasets


class LIME:
    def __init__(self, discretize_continuous=True, discretizer='quartile'):
        """

        :param discretize_continuous: if True, all non-categorical features will be discretized into quartiles.
        :param discretizer: only matters if discretize_continuous is True and data is not sparse.
        Options are 'quartile', 'decile', 'entropy' or a BaseDiscretizer instance.
        """
        self.discretize_continuous = discretize_continuous
        self.discretizer = discretizer

        self.dim = None
        self.ano_idx = None
        return

    def fit(self, x, y, ano_class=1):
        self.ano_idx = np.where(y == 1)[0]
        ano_idx = self.ano_idx
        self.dim = x.shape[1]
        svm = sklearn.svm.SVC(kernel="rbf", probability=True)
        svm.fit(x, y)

        y_pred = svm.predict(x)
        print("Clf model accuracy: [{:.4f}]".format(sklearn.metrics.accuracy_score(y, y_pred)))

        explainer = lime.lime_tabular.LimeTabularExplainer(x, discretize_continuous=self.discretize_continuous,
                                                           discretizer=self.discretizer)
        ano_f_weights = np.zeros([len(ano_idx), self.dim])

        print(len(ano_idx))

        for ii in tqdm(range(len(ano_idx))):
            idx = ano_idx[ii]
            exp = explainer.explain_instance(x[idx], svm.predict_proba, labels=(ano_class,), num_features=self.dim)
            tuples = exp.as_map()[1]
            for tuple in tuples:
                f_id, weight = tuple
                ano_f_weights[ii][f_id] = weight
        return ano_f_weights



