import sklearn
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from alibi.explainers import IntegratedGradients
from tqdm import tqdm
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers


class IntGrad:
    def __init__(self, n_steps=50, method="gausslegendre"):
        """

        :param n_steps:
        :param method:
        """
        self.clf_batch_size = 64
        self.clf_epochs = 30

        self.n_steps = n_steps
        self.method = method

        self.ano_idx = None
        self.nor_idx = None

        self.dim = None
        return

    def fit(self, x, y):
        self.dim = x.shape[1]
        x = min_max_normalize(x)
        # x = z_score_normalize(x)
        y_oh = to_categorical(y, 2)
        clf = self.nn_model()
        clf.fit(x, y_oh, batch_size=self.clf_batch_size, epochs=self.clf_epochs, verbose=1)
        y_pred = clf(x).numpy().argmax(axis=1)
        print("Clf model accuracy: [{:.4f}]".format(sklearn.metrics.accuracy_score(y, y_pred)))

        # Initialize IntegratedGradients instance
        ig = IntegratedGradients(clf, n_steps=self.n_steps, method=self.method)

        # Calculate attributions for the first 10 images in the test set
        self.ano_idx = np.where(y == 1)[0]
        x_ano = x[self.ano_idx]
        # predictions = clf(x_ano).numpy().argmax(axis=1)
        predictions = np.ones(len(self.ano_idx), dtype=int)

        self.nor_idx = np.where(y == 0)[0]
        x_nor = x[self.nor_idx]
        x_nor_avg = np.average(x_nor, axis=0)
        baselines = np.array([x_nor_avg] * len(self.ano_idx))
        explanation = ig.explain(x_ano, baselines=baselines, target=predictions)

        fea_weight_lst = explanation.data['attributions']
        return fea_weight_lst

    def nn_model(self):
        x_in = Input(shape=(self.dim,))
        x = Dense(10, activation='relu')(x_in)
        # x = Dense(10, activation='relu')(x)
        x_out = Dense(2, activation='softmax')(x)
        nn = Model(inputs=x_in, outputs=x_out)
        sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        nn.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        return nn


def min_max_normalize(x):
    n, dim = x.shape
    x_n = np.zeros(x.shape)
    for i in range(dim):
        array = x[:, i]
        _min, _max = np.min(array), np.max(array)
        if _min == _max:
            x_n[:, i] = np.zeros(n)
        else:
            x_n[:, i] = (array - _min) / (_max - _min)

    return x_n


def z_score_normalize(x):
    n, dim = x.shape
    x_n = np.zeros(x.shape)
    for i in range(dim):
        array = x[:, i]
        avg = np.average(array)
        std = np.std(array)
        if std != 0:
            x_n[:, i] = (array - avg) / std
        else:
            x_n[:, i] = array
    return x_n


# import pandas as pd
# path = "../data/00-pima.csv"
# df = pd.read_csv(path)
# X = df.values[:, :-1]
# y = np.array(df.values[:, -1], dtype=int)
# model = IntGrad()
# fw = model.fit(X, y)
