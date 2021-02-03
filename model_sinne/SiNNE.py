import numpy as np
from sklearn.neighbors import NearestNeighbors
from model_sinne import utils
from tqdm import tqdm
import pandas as pd


class SiNNE:
    def __init__(self, max_level="full", width=10, ensemble_num=100, sample_num=8, pretrain=False, verbose=False):
        self.max_level = max_level
        self.width = width
        self.t = ensemble_num
        self.phi = sample_num

        self.pretrain = pretrain
        self.pretrain_nn_model_map = None
        self.verbose = verbose

        return

    def fit(self, x, y):
        dim = x.shape[1]
        norm_idx = np.where(y == 0)[0]
        anom_idx = np.where(y == 1)[0]
        anom_x = x[anom_idx]
        norm_x = x[norm_idx]

        if self.pretrain:
            self.pretrain_nn_model_map = self.pretrain_nn_models_pwset(x, y)
            nn_model_map = self.pretrain_nn_model_map
        else:
            nn_model_map = {}

        if self.max_level == "full":
            self.max_level = dim
        else:
            self.max_level = int(self.max_level)

        # step1: get the scores of dim=1
        dim1_scores_lst = np.zeros([dim, len(anom_idx)])
        if self.verbose:
            print("Running level 1: ")
        for i in tqdm(range(dim)):
            if self.pretrain:
                [ensmb_samples_lst, ensmb_radius_lst, _] = nn_model_map[str([i])]
            else:
                norm_x_subspace = norm_x[:, [i]]
                [ensmb_samples_lst, ensmb_radius_lst, _] = self.training_nn_models(norm_x_subspace)

            subspaced_queries = anom_x[:, [i]]
            # dim1_scores_lst[i] = scoring(subspaced_queries, ensmb_radius_lst, ensmb_nn_lst)
            for jj, q in enumerate(subspaced_queries):
                dim1_scores_lst[i, jj] = self.single_scoring(q, ensmb_samples_lst, ensmb_radius_lst)

        D = np.arange(dim)
        exp_subspace_lst = []
        for i in tqdm(range(len(anom_idx))):
            query = anom_x[i]

            init_score = dim1_scores_lst[:, i]
            init_subspaces = [[i] for i in range(dim)]

            if dim <= 50:
                keep_score = init_score
                keep_subspaces = init_subspaces
            else:
                start = len(init_score) - self.width
                keep_score = np.sort(init_score)[start:]
                indices = np.argsort(init_score)[start:]
                keep_subspaces = [init_subspaces[dd] for dd in indices]

            for level in range(2, self.max_level):
                if self.verbose:
                    print("--------------------- level: [{}] ----------------------".format(level))

                # filter the subspaces that are in previous level (has been explored)
                root_subspaces = [s for s in keep_subspaces if len(s) == level - 1]
                exploring_subspaces = []
                for s in root_subspaces:
                    other_features = np.setdiff1d(D, s)
                    for f in other_features:
                        this_subspace = list(np.sort(s + [f]))
                        if this_subspace not in exploring_subspaces:
                            exploring_subspaces.append(this_subspace)
                            # print("add to exploring set")
                if self.verbose:
                    print("exploring subspaces size: ", len(exploring_subspaces))
                exploring_scores = np.zeros(len(exploring_subspaces))
                if self.verbose:
                    iterator = tqdm(range(len(exploring_subspaces)))
                else:
                    iterator = range(len(exploring_subspaces))

                for jj in iterator:
                    s = exploring_subspaces[jj]
                    if self.pretrain or str(s) in nn_model_map:
                        [ensmb_samples_lst, ensmb_radius_lst, ensmb_nn_lst] = nn_model_map[str(s)]
                    else:
                        norm_x_subspace = norm_x[:, s]
                        nn_model = self.training_nn_models(norm_x_subspace)
                        nn_model_map[str(s)] = nn_model
                        [ensmb_samples_lst, ensmb_radius_lst, ensmb_nn_lst] = nn_model

                    # subspaced_query = [query[s]]
                    # query_subspace_score = scoring(subspaced_query, ensmb_radius_lst, ensmb_nn_lst)

                    # @NOTE: use a small bias to get larger score for shorter subspace,
                    # then the model tend to use shorter subspaces as explanation if multiple subspaces have same score
                    query_subspace_score = self.single_scoring(query[s], ensmb_samples_lst, ensmb_radius_lst) + \
                                           (dim - len(s)) * 0.001
                    exploring_scores[jj] = query_subspace_score


                scores = np.append(keep_score, exploring_scores)
                subspaces = keep_subspaces + exploring_subspaces

                if self.width > len(scores):
                    start = 0
                else:
                    start = len(scores) - self.width
                keep_score = np.sort(scores)[start:]
                indices = np.argsort(scores)[start:]
                keep_subspaces = [subspaces[dd] for dd in indices]

                if self.verbose:
                    print("--------------------- level: [{}] ----------------------".format(level))
                    print(keep_score)
                    print(keep_subspaces)
            exp_subspace = keep_subspaces[-1]
            exp_subspace_lst.append(exp_subspace)
        return exp_subspace_lst

    def training_nn_models(self, data):
        n_x = data.shape[0]
        ensmb_samples_lst = []
        ensmb_radius_lst = []
        ensmb_nn_lst = []
        for i in range(self.t):
            samples = data[np.random.choice(np.arange(n_x), self.phi, replace=False)]
            ensmb_samples_lst.append(samples)

            # the nearest neighbor is data itself, so the n_neighbors is set as 2
            samples_nn = NearestNeighbors(n_neighbors=2).fit(samples)
            ensmb_nn_lst.append(samples_nn)

            radius = np.zeros(self.phi)
            for ii, xx in enumerate(samples):
                # nbr_idx = nbrs_local.kneighbors([xx])[1].flatten()[1]
                radius[ii] = samples_nn.kneighbors([xx])[0].flatten()[1]
            ensmb_radius_lst.append(radius)

        nn_model = [ensmb_samples_lst, ensmb_radius_lst, ensmb_nn_lst]
        return nn_model

    def single_scoring(self, single_x, ensmb_samples_lst, ensmb_radius_lst):
        outlier_score = 0

        for i in range(self.t):
            radius = ensmb_radius_lst[i]
            samples = ensmb_samples_lst[i]

            is_outlier = 1
            for j in range(self.phi):
                sample = samples[j]
                threshold = radius[j]
                dist = np.sqrt(np.sum((sample - single_x)**2))
                if dist <= threshold:
                    is_outlier = 0
                    break
            outlier_score += is_outlier

        outlier_score = outlier_score / self.t
        return outlier_score

    # @TODO bug: it is wrong to only consider the nearest sample data in each model
    def scoring(self, test_x, ensmb_radius_lst, ensmb_nn_lst):
        outlier_scores = np.zeros(len(test_x))
        t = len(ensmb_radius_lst)
        num_x = len(test_x)

        for i in range(t):
            radius = ensmb_radius_lst[i]
            nn = ensmb_nn_lst[i]

            # choosing the nearest data in the model i
            nbr_idx = nn.kneighbors(test_x)[1][:, 0]
            dists = nn.kneighbors(test_x)[0][:, 0]
            thresholds = radius[nbr_idx]

            for j in range(num_x):
                dist = dists[j]
                threshold = thresholds[j]
                if dist <= threshold:
                    outlier_scores[j] += 0
                else:
                    outlier_scores[j] += 1

        outlier_scores = outlier_scores / t
        return outlier_scores

    def pretrain_nn_models_pwset(self, x, y):
        dim = x.shape[1]
        norm_idx = np.where(y == 0)[0]
        x_norm = x[norm_idx]

        full_set = np.arange(dim)
        pwset = utils.powerset(full_set)
        pwset.remove([])
        pwset_nn_model_map = {}

        for subspace in tqdm(pwset):
            norm_x_subspace = x_norm[:, subspace]
            nn_model = self.training_nn_models(norm_x_subspace)
            pwset_nn_model_map[str(subspace)] = nn_model
        return pwset_nn_model_map

    def fit_od(self, x):
        [ensmb_samples_lst, ensmb_radius_lst, _] = self.training_nn_models(x)
        score_lst = []
        for i in tqdm(range(len(x))):
            xx = x[i]
            score = self.single_scoring(xx, ensmb_samples_lst, ensmb_radius_lst)
            score_lst.append(score)
        return score_lst


# if __name__ == '__main__':
#     root = 'E:/1-anomaly detection/10-AnoExp/'
#     path = root + "data/tabular/new_pca/cardio_pca.csv"
#     df = pd.read_csv(path)
#     x = df.values[:, :-1]
#     y = np.array(df.values[:, -1], dtype=int)
#
#     model = SiNNE(max_level="full", width=10, ensemble_num=100, sample_num=8, pretrain=False)
#     # exp_subspace_lst = model.fit(x,y)
#     # precision, jaccard, score = evaluation_od.evaluation_od(exp_subspace_lst, x, y,
#     #                                                         "thyroid", model_name="iforest")
#     # metric_lst = [precision, jaccard, score]
#     # print(metric_lst)
#
#     # norm_idx = np.where(y == 0)[0]
#     # norm_x = x[norm_idx]
#     # [ensmb_samples_lst, ensmb_radius_lst, ensmb_nn_lst] = model.training_nn_models(norm_x)
#     # score_lst = []
#     # for i in tqdm(range(len(x))):
#     #     xx = x[i]
#     #     score = model.single_scoring(xx, ensmb_samples_lst, ensmb_radius_lst)
#     #     score_lst.append(score)
#     # from sklearn import metrics
#     # print(metrics.roc_auc_score(y, score_lst))