import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.neighbors import NearestNeighbors


class SingleTripletDataset(Dataset):
    def __init__(self, anom_idx, x, y, triplets_selector, transform=None):
        self.transform = transform
        self.data = x
        self.triplets = triplets_selector.get_triplets(anom_idx, x, y)

    def __getitem__(self, index):
        a_idx, p_idx, n_idx = self.triplets[index]
        anchor, positive, negative = self.data[a_idx], self.data[p_idx], self.data[n_idx]
        if self.transform is not None:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)
        return anchor, positive, negative

    def __len__(self):
        return len(self.triplets)


class SingleDataset(Dataset):
    def __init__(self, anom_idx, x, y, data_selector, transform=None):
        self.transform = transform
        self.selected_data = data_selector.get_data(anom_idx, x, y)

    def __getitem__(self, index):
        data = self.selected_data[0][index]
        target = self.selected_data[1][index]
        if self.transform is not None:
            data = self.transform(data)
        return data, target

    def __len__(self):
        return len(self.selected_data[0])


class SingleTripletDatasetClf(Dataset):
    def __init__(self, anom_idx, x, y, triplets_selector, transform=None):
        self.transform = transform
        self.data = x
        self.triplets, self.targets = triplets_selector.get_triplets(anom_idx, x, y)

    def __getitem__(self, index):
        a_idx, p_idx, n_idx = self.triplets[index]
        a_target, p_target, n_target = self.targets[index]
        anchor, positive, negative = self.data[a_idx], self.data[p_idx], self.data[n_idx]
        if self.transform is not None:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)
        return anchor, positive, negative, a_target, p_target, n_target

    def __len__(self):
        return len(self.triplets)


class MyHardSingleTripletSelector:
    def __init__(self, nbrs_num, rand_num, nbr_indices):
        self.x = None
        self.y = None
        self.nbrs_num = nbrs_num
        self.rand_num = rand_num
        self.nbr_indices = nbr_indices

    def get_triplets(self, anom_idx, x, y, normal_label=0):
        self.x = x.cpu().data.numpy()
        self.y = y.cpu().data.numpy()

        # anom_x = self.x[anom_idx]
        # x_noml = self.x[noml_idx]
        # n_neighbors = self.nbrs_num
        # nbrs_local = NearestNeighbors(n_neighbors=n_neighbors).fit(x_noml)
        # nbr_indices = noml_idx[nbrs_local.kneighbors([anom_x])[1].flatten()]

        noml_idx = np.where(self.y == normal_label)[0]
        nbr_indices = self.nbr_indices
        rand_num = self.rand_num

        rand_canddt = np.setdiff1d(noml_idx, nbr_indices)
        rand_indices = np.random.choice(rand_canddt, rand_num, replace=False)

        triplets = [[anchor, positive, anom_idx]
                    for anchor in rand_indices
                    for positive in nbr_indices]
        return torch.LongTensor(np.array(triplets))


class MyHardSingleSelectorClf:
    def __init__(self, nbrs_num, rand_num):
        self.nbrs_num = nbrs_num
        self.rand_num = rand_num

    def get_data(self, anom_idx, x, y, normal_label=0):
        x = x.cpu().data.numpy()
        y = y.cpu().data.numpy()

        anom_x = x[anom_idx]
        noml_idx = np.where(y == normal_label)[0]
        x_noml = x[noml_idx]

        nbrs_local = NearestNeighbors(n_neighbors=self.nbrs_num).fit(x_noml)
        nbr_indices = noml_idx[nbrs_local.kneighbors([anom_x])[1].flatten()]
        rand_canddt = np.setdiff1d(noml_idx, nbr_indices)
        rand_indices = np.random.choice(rand_canddt, self.rand_num, replace=False)

        # perturbation to augment
        dim = x.shape[1]
        anom_lst = []
        anom_lst.append(anom_x)
        for i in range(self.rand_num + self.nbrs_num -1):
            new_anom_x = anom_x.copy()
            choose_f = np.random.choice(np.arange(dim), 3)
            for a in choose_f:
                new_anom_x[a] = anom_x[a] * 1.01
            anom_lst.append(new_anom_x)

        data_idx = np.hstack([rand_indices, nbr_indices])
        norm_data = x[data_idx]
        data = np.vstack([np.array(anom_lst), norm_data])
        target = np.hstack([np.ones(10), np.zeros(len(rand_indices), dtype=int), np.zeros(len(nbr_indices), dtype=int)])

        return torch.FloatTensor(data), torch.LongTensor(target)


class MyHardSingleTripletSelectorClf:
    def __init__(self, nbrs_num, rand_num):
        self.x = None
        self.y = None
        self.nbrs_num = nbrs_num
        self.rand_num = rand_num

    def get_triplets(self, anom_idx, x, y, normal_label=0):
        self.x = x.cpu().data.numpy()
        self.y = y.cpu().data.numpy()

        anom_x = self.x[anom_idx]
        noml_idx = np.where(self.y == normal_label)[0]
        x_noml = self.x[noml_idx]
        n_neighbors = self.nbrs_num
        rand_num = self.rand_num

        nbrs_local = NearestNeighbors(n_neighbors=n_neighbors).fit(x_noml)

        nbr_indices = noml_idx[nbrs_local.kneighbors([anom_x])[1].flatten()]
        # nbr_dist = nbrs_local.kneighbors([anom_x])[0].flatten()

        rand_canddt = np.setdiff1d(noml_idx, nbr_indices)
        rand_indices = np.random.choice(rand_canddt, rand_num, replace=False)

        triplets = [[anchor, positive, anom_idx]
                    for anchor in rand_indices
                    for positive in nbr_indices]

        # print("Generate triplets Num: [%d]" % len(triplets))
        target = [[0, 0, 1]] * len(triplets)

        return torch.LongTensor(np.array(triplets)), torch.LongTensor(np.array(target))


class MyHardSingleTripletSelector2:
    def __init__(self, nbrs_num, rand_num):
        self.x = None
        self.y = None
        self.nbrs_num = nbrs_num
        self.rand_num = rand_num

    def get_triplets(self, anom_idx, x, y, normal_label=0):
        self.x = x.cpu().data.numpy()
        self.y = y.cpu().data.numpy()

        n_neighbors = self.nbrs_num
        rand_num = self.rand_num

        anom_x = self.x[anom_idx]

        anom_indices = np.where(self.y != normal_label)[0]
        noml_indices = np.where(self.y == normal_label)[0]
        noml_x = self.x[noml_indices]
        
        nbrs_local = NearestNeighbors(n_neighbors=n_neighbors).fit(noml_x)
        nbr_indices = noml_indices[nbrs_local.kneighbors([anom_x])[1].flatten()]
        # nbr_dist = nbrs_local.kneighbors([anom_x])[0].flatten()

        rand_canddt_nor = np.setdiff1d(noml_indices, nbr_indices)
        rand_nor_indices = np.random.choice(rand_canddt_nor, rand_num, replace=False)

        triplets1 = [[anchor, positive, anom_idx]
                     for anchor in rand_nor_indices
                     for positive in nbr_indices]
        
        rand_canddt_ano = np.setdiff1d(anom_indices, anom_idx)
        if len(rand_canddt_ano) < rand_num:
            rand_ano_indices = rand_canddt_ano
        else:
            rand_ano_indices = np.random.choice(rand_canddt_ano, rand_num, replace=False)

        triplets2 = [[anchor, anom_idx, negative]
                     for anchor in rand_ano_indices
                     for negative in nbr_indices]
        triplets = triplets1 + triplets2

        # print("Generate triplets Num: [%d]" % len(triplets))
        target1 = [[0, 0, 1]] * len(triplets1)
        target2 = [[1, 1, 0]] * len(triplets2)
        target = target1 + target2

        return torch.LongTensor(np.array(triplets)), torch.LongTensor(np.array(target))

