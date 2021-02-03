from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np

classifiers = {
    'knn': KNeighborsClassifier(),
    'rbf_svm': SVC(probability=True),
    'adaBoost': AdaBoostClassifier(),
    'gradient_boosting': GradientBoostingClassifier(),
    'bagging': BaggingClassifier(),
    'naive_bayes': GaussianNB(),
    'mlp': MLPClassifier()
}


# for name in classifiers.keys():
#     clf = classifiers[name]


def get_clf_model_names():
    return classifiers.keys()


def train_clf(x_train, y_train, clf_model="decision_tree"):
    """
    train a classification model
    :param x_train: np array
    :param y_train:
    :param clf_model:
    :return:
    """
    clf = classifiers[clf_model]
    clf.fit(x_train, y_train)
    return clf


def test_clf(x_test, clf):
    """

    :param x_test:
    :param clf:
    :return: y_pred: array, shape [n_samples]; Predicted class label per sample.
    """
    y_pre = clf.predict(x_test)
    y_prob_lst = clf.predict_proba(x_test)
    y_prob = np.array([max(prob) for prob in y_prob_lst])
    return y_pre, y_prob

