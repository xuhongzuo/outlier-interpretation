import numpy as np
import time, math
import torch
import torch.utils.data as Data
import torch.optim as optim
import torch.nn.functional as F

from torch.optim import lr_scheduler
from sklearn import metrics
from tqdm import tqdm
from model_aton.utils import EarlyStopping, min_max_normalize

from model_aton.datasets import MyHardSingleSelectorClf, SingleDataset
from model_aton.networks import ATONabla3net, AttentionNet, ClassificationNet
import warnings
warnings.filterwarnings("ignore")


class ATONabla3:
    def __init__(self, nbrs_num=30, rand_num=30,
                 n_epoch=10, batch_size=64, lr=0.1, n_linear=64, margin=2.,
                 verbose=True, gpu=True):
        self.verbose = verbose

        self.x = None
        self.y = None
        self.ano_idx = None
        self.dim = None

        self.reason_map = {}

        cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if cuda and gpu else "cpu")
        if cuda:
            torch.cuda.set_device(0)
        print("device:", self.device)

        self.nbrs_num = nbrs_num
        self.rand_num = rand_num

        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.lr = lr
        self.n_linear = n_linear
        self.margin = margin
        return

    def fit(self, x, y):
        device = self.device
        self.dim = x.shape[1]
        x = min_max_normalize(x)
        self.ano_idx = np.where(y == 1)[0]

        self.x = torch.tensor(x, dtype=torch.float32).to(device)
        self.y = torch.tensor(y, dtype=torch.int64).to(device)

        # train model for each anomaly
        attn_lst, W_lst = [], []
        if self.verbose:
            iterator = range(len(self.ano_idx))
        else:
            iterator = tqdm(range(len(self.ano_idx)))
        for ii in iterator:
            idx = self.ano_idx[ii]

            s_t = time.time()
            attn, W = self.interpret_ano(idx)
            attn_lst.append(attn)
            W_lst.append(W)

            if self.verbose:
                print("ano_idx [{} ({})] attn: {}".format(ii, idx, attn))
                print("Ano_id:[{}], ({}/{}) \t time: {:.2f}s\n".format(
                    idx, (ii + 1), len(self.ano_idx),
                    (time.time() - s_t)))

        fea_weight_lst = []
        for ii, idx in enumerate(self.ano_idx):
            attn, w = attn_lst[ii], W_lst[ii]
            fea_weight = np.zeros(self.dim)

            # attention (linear space) + w --> feature weight (original space)
            for j in range(len(attn)):
                fea_weight += attn[j] * abs(w[j])
            fea_weight_lst.append(fea_weight)
        return fea_weight_lst

    def interpret_ano(self, idx):
        device = self.device
        dim = self.dim

        data_loader, test_loader = self.prepare_triplets(idx)
        n_linear = self.n_linear
        attn_net = AttentionNet(in_feature=n_linear, n_hidden=int(1.5 * n_linear), out_feature=n_linear)
        clf_net = ClassificationNet(n_feature=n_linear)

        model = ATONabla3net(attn_net=attn_net, clf_net=clf_net, n_feature=dim, n_linear=n_linear)
        model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-2)
        criterion_cel = torch.nn.CrossEntropyLoss()

        scheduler = lr_scheduler.StepLR(optimizer, 5, gamma=0.1)
        early_stp = EarlyStopping(patience=3, verbose=False)

        for epoch in range(self.n_epoch):
            model.train()
            total_loss = 0
            total_acc = 0
            es_time = time.time()

            batch_cnt = 0
            for batch_x, batch_y in data_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                clf_out, attn = model(batch_x)
                loss = criterion_cel(clf_out, batch_y)

                _, y_pred = torch.max(F.softmax(clf_out, dim=1).data.cpu(), 1)
                clf_acc = metrics.accuracy_score(batch_y.cpu().data.numpy(), y_pred.cpu().data.numpy())

                total_loss += loss
                total_acc += clf_acc

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_cnt += 1

            train_loss = total_loss / batch_cnt
            train_acc = total_acc / batch_cnt
            est = time.time() - es_time

            if self.verbose and (epoch + 1) % 1 == 0:
                print('Epoch: [{:02}/{:02}]  loss: {:.4f} acc: {:.4f} Time: {:.2f}s'
                      .format(epoch + 1, self.n_epoch, train_loss, train_acc,  est))
            scheduler.step()

            early_stp(train_loss, model)
            if early_stp.early_stop:
                model.load_state_dict(torch.load(early_stp.path))
                if self.verbose:
                    print("early stopping")
                break

        for x, target in test_loader:
            model.eval()
            x = x.to(device)
            _, attn = model(x)

        attn_avg = torch.mean(attn, dim=0)
        attn_avg = attn_avg.data.cpu().numpy()
        W = model.linear.weight.data.cpu().numpy()
        return attn_avg, W

    def prepare_triplets(self, idx):
        x = self.x
        y = self.y

        selector = MyHardSingleSelectorClf(nbrs_num=self.nbrs_num, rand_num=self.rand_num)
        dataset = SingleDataset(idx, x, y, data_selector=selector)

        data_loader = Data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = Data.DataLoader(dataset, batch_size=len(dataset))
        return data_loader, test_loader

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
