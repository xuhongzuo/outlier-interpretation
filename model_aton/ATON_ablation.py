import numpy as np
import time, math
import torch
import torch.utils.data as Data
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm

from model_aton.utils import EarlyStopping, min_max_normalize
from model_aton.datasets import MyHardSingleTripletSelector
from model_aton.datasets import SingleTripletDataset
from model_aton.networks import ATONablanet


class ATONabla:
    """
    ablated version that removes self-attention mechanism
    """
    def __init__(self, nbrs_num=30, rand_num=30,
                 n_epoch=10, batch_size=64, lr=0.1, n_linear=64, margin=5.,
                 verbose=True):
        self.verbose = verbose

        self.x = None
        self.y = None
        self.ano_idx = None
        self.dim = None

        self.reason_map = {}

        cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if cuda else "cpu")
        if cuda:
            torch.cuda.set_device(0)

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

        W_lst = []
        if self.verbose:
            iterator = range(len(self.ano_idx))
        else:
            iterator = tqdm(range(len(self.ano_idx)))
        for ii in iterator:
            idx = self.ano_idx[ii]

            s_t = time.time()
            W = self.interpret_ano(idx)
            W_lst.append(W)
            if self.verbose:
                print("Ano_id:[{}], ({}/{}) \t time: {:.2f}s\n".format(
                    idx, (ii + 1), len(self.ano_idx), (time.time() - s_t)))

        fea_weight_lst = []
        for ii, idx in enumerate(self.ano_idx):
            w = W_lst[ii]
            fea_weight = np.zeros(self.dim)
            for j in range(len(w)):
                fea_weight += abs(w[j])
            fea_weight_lst.append(fea_weight)
        return fea_weight_lst

    def interpret_ano(self, idx):
        device = self.device
        dim = self.dim

        data_loader, test_loader = self.prepare_triplets(idx)
        n_linear = self.n_linear
        model = ATONablanet(n_feature=dim, n_linear=n_linear)
        model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-2)
        criterion_tml = torch.nn.TripletMarginLoss(margin=self.margin, p=2)

        scheduler = lr_scheduler.StepLR(optimizer, 5, gamma=0.1)
        early_stp = EarlyStopping(patience=3, verbose=False)

        for epoch in range(self.n_epoch):
            model.train()
            total_loss = 0
            es_time = time.time()

            batch_cnt = 0
            for anchor, pos, neg in data_loader:
                anchor, pos, neg = anchor.to(device), pos.to(device), neg.to(device)
                embed_anchor, embed_pos, embed_neg = model(anchor, pos, neg)
                loss = criterion_tml(embed_anchor, embed_pos, embed_neg)

                total_loss += loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_cnt += 1

            train_loss = total_loss / batch_cnt
            est = time.time() - es_time
            if (epoch + 1) % 1 == 0 and self.verbose:
                message = 'Epoch: [{:02}/{:02}]  loss: {:.4f} Time: {:.2f}s'.format(
                    epoch + 1, self.n_epoch,
                    train_loss, est)
                print(message)
            scheduler.step()

            early_stp(train_loss, model)
            if early_stp.early_stop:
                model.load_state_dict(torch.load(early_stp.path))
                if self.verbose:
                    print("early stopping")
                break

        W = model.linear.weight.data.cpu().numpy()
        return W

    def prepare_triplets(self, idx):
        x = self.x
        y = self.y
        selector = MyHardSingleTripletSelector(nbrs_num=self.nbrs_num, rand_num=self.rand_num)
        dataset = SingleTripletDataset(idx, x, y, triplets_selector=selector)
        data_loader = Data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = Data.DataLoader(dataset, batch_size=len(dataset))
        return data_loader, test_loader

