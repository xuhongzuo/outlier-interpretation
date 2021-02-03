import os
import copy
import math
import time
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn import svm
from model_coin.prediction_strength import optimalK
from tqdm import tqdm


class COIN(object):
    def __init__(self, data, inds_otlr, nbrs_ratio,
                 AUG=1.0, MIN_CLUSTER_SIZE=5, MAX_NUM_CLUSTER=4, VAL_TIMES=10, C_SVM=1.,
                 RESOLUTION=0.05, THRE_PS=0.85, DEFK=0):
        """
            data:  Data matrix, each row represents one instance
            inds_otlr:  A vector with each entry telling whether this instance is outlier (1) or not (0)
            nbrs_ratio:  The ratio of normal instances as the context for each outlier
            AUG: An additional feature attached to the input as data augmentation
            MIN_CLUSTER_SIZE: Minimum number of nodes in each cluster
            MAX_NUM_CLUSTER: Maximum number of clusters considered in prediction strength computation
            VAL_TIMES: Number of iterations for computing prediction strength
            C_SVM: A hyperparameter in SVM (optimum value would be better to be estimated through validation)
            DEFK: Predefined number of clusters in each context. Value 0 means using Prediction Strength to estimate it.
        """
        self.data = data
        self.dim = data.shape[1]

        self.inds_otlr = inds_otlr
        self.ano_idx = np.where(inds_otlr == 1)[0]

        self.AUG = float(AUG)

        self.num_inst = data.shape[0]
        self.num_feat = data.shape[1]
        self.num_nbrs = int(nbrs_ratio * self.num_inst)

        self.MIN_CLUSTER_SIZE = MIN_CLUSTER_SIZE
        self.MAX_NUM_CLUSTER = MAX_NUM_CLUSTER
        self.VAL_TIMES = VAL_TIMES
        self.C_SVM = C_SVM
        self.RESOLUTION = RESOLUTION
        self.THRE_PS = THRE_PS
        self.DEFK = DEFK

        # normal instances
        self.data_normal = self.data[np.where(self.inds_otlr == 0)[0]]

        # nearest nbrs object based on normal instances
        self.nbrs = NearestNeighbors(n_neighbors=self.num_nbrs, n_jobs=-1)
        self.nbrs.fit(self.data_normal)

    def interpret_outliers(self, ids_target, sgnf_vec, int_flag=0):
        """
            ids_target: Indices of target outliers
            sgnf_vec: A vector indicating the importance of each attribute, as prior knowledge
            int_flag: Discrete attribute or not
            :return: A list of sorted (outlier_ID, outlierness) tuples, a list of clfs, attr importance 2D-array
        """

        # Attach 0 to the augmented feature
        if isinstance(sgnf_vec, int) or isinstance(sgnf_vec, float):
            sgnf_vec = np.hstack((np.ones(self.num_feat), 0))
        else:
            sgnf_vec = np.hstack((sgnf_vec, [0]))

        # Interpret each target outlier
        oid_devt_dict = dict()  # id-score tuples
        score_attr_mat = []

        for ii in tqdm(range(len(ids_target))):
            i = ids_target[ii]

            # Do clustering on the context, build one classifier for each cluster
            nums_c, clfs, cluster_attr_scale = self.cluster_context(i, int_flag)

            # Calculate outlierness score
            devt_i = self.CalculateOutlierness(i, clfs, nums_c, sgnf_vec)
            oid_devt_dict[i] = devt_i

            # Find outlying attributes
            score_attr = np.zeros(self.num_feat)
            for num_c, clf in zip(nums_c, clfs):
                score_attr += num_c * np.abs(clf.coef_[0])      # weighted by the normal cluster size
            score_attr /= float(np.sum(nums_c))
            score_attr /= np.sum(score_attr)    # relative importance
            score_attr_mat.append(copy.copy(score_attr))
            # print(score_attr)

        return np.array(score_attr_mat), oid_devt_dict

    def cluster_context(self, id_outlier, int_flag):
        # find the context of the outlier
        dist_btwn, otlr_nbrs = self.nbrs.kneighbors([self.data[id_outlier]])
        dist_btwn, otlr_nbrs = dist_btwn[0], self.data_normal[otlr_nbrs[0], :]
        # print(self.data[id_outlier])
        # print(otlr_nbrs)

        # choose the number of clusters in the context
        if self.DEFK == 0:
            k_best = optimalK(otlr_nbrs, self.VAL_TIMES, self.MAX_NUM_CLUSTER, self.THRE_PS)
        else:
            k_best = self.DEFK
        k_best = min(k_best+1, self.MAX_NUM_CLUSTER)     # empirically, it is better to have a lager K
        # print('Best k:', k_best)

        # clutering the context
        kmeans = KMeans(n_clusters=k_best, random_state=0).fit(otlr_nbrs)
        label_nbrs = kmeans.labels_

        clfs = []
        nbrs_mean = []
        nums_c = []
        cluster_attr_scale = []

        # build a linear classifier for each cluster of nbrs
        for c in range(k_best):
            # indices for instances in cluster c
            inds_c = np.where(label_nbrs == c)[0]

            # the cluster cannot be too small
            if np.size(inds_c) < self.MIN_CLUSTER_SIZE:
                continue
            nums_c.append(len(inds_c))

            # instances for cluster c
            otlr_nbrs_c = otlr_nbrs[inds_c, :]
            dist_btwn_c = dist_btwn[inds_c]

            # distance property of cluster c
            cluster_attr_scale.append(np.hstack((np.max(otlr_nbrs_c, axis=0) - np.min(otlr_nbrs_c, axis=0), 0)))  # scale for each attr

            # synthetic sampling to build two classes
            insts_c0 = self.SyntheticSampling(otlr_nbrs_c, self.data[id_outlier], int_flag)
            insts_c1 = otlr_nbrs_c

            clf = self.SVCInterpreter(insts_c0, insts_c1)
            clfs.append(clf)
            nbrs_mean.append(np.average(insts_c1, axis=0))

        return nums_c, clfs, cluster_attr_scale

    def SyntheticSampling(self, insts, otlr, int_flag):
        '''
        Expand the outlier into a class.

        insts: normal instances
        otlr: the outlier instance
        expand_ratio: expand ratio
        int_flag: whether to round to int
        :return: two classes of data points
        '''

        num_c0_new = insts.shape[0] - 1
        coeff_c0_new = np.random.rand(num_c0_new, insts.shape[0])   # transformation matrix for synthetic sampling
        nbrs_local = NearestNeighbors(n_neighbors=1).fit(insts)
        min_dist_to_nbr = nbrs_local.kneighbors([otlr])[0][0, 0]/insts.shape[1]

        for r in range(coeff_c0_new.shape[0]):
            coeff_c0_new[r, :] /= sum(coeff_c0_new[r, :])
        insts_c0_new = np.dot(coeff_c0_new, insts - np.dot(np.ones((insts.shape[0], 1)), [otlr]))
        for r in range(insts_c0_new.shape[0]):                      # shrink to prevent overlap
            insts_c0_new[r, :] *= (0.2 * np.random.rand(1)[0] * min_dist_to_nbr)
        insts_c0_new += np.dot(np.ones((num_c0_new, 1)), [otlr])    # origin + shift
        if int_flag:
            insts_c0_new = np.round(insts_c0_new)
        insts_c0 = np.vstack((otlr, insts_c0_new))

        return insts_c0

    def SVCInterpreter(self, insts_c0, insts_c1):
        # classification between normal instances and outliers, where outliers have negative output

        clf = svm.LinearSVC(penalty='l1', C=self.C_SVM, dual=False, intercept_scaling=self.AUG)
        X_c = np.vstack((insts_c0, insts_c1))
        y_c = np.hstack((np.zeros(insts_c0.shape[0]), np.ones(insts_c1.shape[0])))
        clf.fit(X_c, y_c)
        #print(insts_c1)
        #print(insts_c0)

        return clf

    def CalculateOutlierness(self, id_outlier, clfs, nums_c, sgnf_vec):
        otlr = self.data[id_outlier]

        devt_overall = 0.
        for c in range(len(nums_c)):
            # distance to the boundary
            otlr_aug = np.hstack((otlr, self.AUG))
            w = np.hstack((clfs[c].coef_[0], clfs[c].intercept_[0]/self.AUG))
            w_a = np.hstack((clfs[c].coef_[0], 0))
            dist = -min(0, np.inner(otlr_aug, w))/np.linalg.norm(w_a)

            # rescale deviation according to attributes' importance
            devt = np.linalg.norm(np.multiply(dist * w_a / np.linalg.norm(w_a), sgnf_vec))
            if np.isnan(devt):
                devt = 0.

            # weighted by the opponent cluster size
            devt_overall += devt * nums_c[c]

        devt_overall /= sum(nums_c)

        return devt_overall

    def fit(self, sgnf_prior):
        importance_attr, outlierness = self.interpret_outliers(self.ano_idx, sgnf_prior)
        return importance_attr

    def weight2subspace(self, weight, r=0.7, num=-1):
        threshold = r * np.sum(weight)
        tmp_s = 0
        exp_subspace = []
        sorted_idx1 = np.argsort(weight)
        sorted_idx = [sorted_idx1[self.dim - i -1] for i in range(self.dim)]
        if num != -1:
            exp_subspace = sorted_idx[:num]
            exp_subspace = list(np.sort(exp_subspace))
            return exp_subspace

        for idx in sorted_idx:
            tmp_s += weight[idx]
            exp_subspace.append(idx)
            if tmp_s >= threshold:
                break
        exp_subspace = list(np.sort(exp_subspace))
        return exp_subspace

    def weight2subspace_pn(self, weight):
        exp_subspace = []
        for i in range(len(weight)):
            # exp_subspace.append(list(np.where(weight[i] > 0)[0]))
            if weight[i] > 0:
                exp_subspace.append(i)
        if len(exp_subspace) == 0:
            exp_subspace = np.arange(len(weight))
        exp_subspace = list(np.sort(exp_subspace))
        return exp_subspace

    def get_exp_subspace(self, fea_weight_lst, w2s_ratio, real_exp_len=None):
        exp_subspace_lst = []
        for ii, idx in enumerate(self.ano_idx):
            fea_weight = fea_weight_lst[ii]
            if w2s_ratio == "real_len":
                exp_subspace_lst.append(self.weight2subspace(fea_weight, num=real_exp_len[ii]))
            elif w2s_ratio == "auto":
                r = math.sqrt(2 / self.dim)
                exp_subspace_lst.append(self.weight2subspace(fea_weight, r=r))
            elif w2s_ratio == "pn":
                exp_subspace_lst.append(self.weight2subspace_pn(fea_weight))
            else:
                exp_subspace_lst.append(self.weight2subspace(fea_weight, r=w2s_ratio))
        return exp_subspace_lst