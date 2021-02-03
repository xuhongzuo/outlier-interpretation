import torch
import torch.nn as nn
import torch.nn.functional as F


class ATONnet(nn.Module):
    def __init__(self, attn_net, n_feature, n_linear):
        super(ATONnet, self).__init__()
        self.attn_net = attn_net
        self.linear = torch.nn.Linear(n_feature, n_linear, bias=False)

    def forward(self, anchor, positive, negative):
        anchor = self.linear(anchor)
        positive = self.linear(positive)
        negative = self.linear(negative)

        cat = torch.cat([negative, anchor, positive], dim=1)

        attn = self.attn_net(cat)
        embedded_n = negative * attn
        embedded_a = anchor * attn
        embedded_p = positive * attn

        embedded_n_dff = (1 - attn) * negative
        embedded_a_dff = (1 - attn) * anchor
        embedded_p_dff = (1 - attn) * positive
        dis1 = F.pairwise_distance(embedded_n_dff, embedded_a_dff)
        dis2 = F.pairwise_distance(embedded_p_dff, embedded_a_dff)
        dis = torch.abs(dis1 - dis2)

        return embedded_a, embedded_p, embedded_n, attn, dis

    def get_lnr(self, x):
        return self.linear(x)


class AttentionNet(nn.Module):
    def __init__(self, in_feature, n_hidden, out_feature):
        super(AttentionNet, self).__init__()
        self.hidden = torch.nn.Linear(in_feature, n_hidden)
        self.out = torch.nn.Linear(n_hidden, out_feature)

    def forward(self, x):
        x = torch.relu(self.hidden(x))
        x = self.out(x)
        _min = torch.unsqueeze(torch.min(x, dim=1)[0], 0).t()
        _max = torch.unsqueeze(torch.max(x, dim=1)[0], 0).t()
        x = (x - _min) / (_max - _min)
        return x


class MyLoss(nn.Module):
    """
    triplet deviation-based loss
    """
    def __init__(self, alpha1, alpha2, margin):
        super(MyLoss, self).__init__()
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.criterion_tml = torch.nn.TripletMarginLoss(margin=margin, p=2)
        return

    def forward(self, embed_anchor, embed_pos, embed_neg, dis):
        loss_tml = self.criterion_tml(embed_anchor, embed_pos, embed_neg)
        loss_dis = torch.mean(dis)
        loss = self.alpha1 * loss_tml + self.alpha2 * loss_dis
        return loss


# ---------------------- ATON - ablation -------------------------- #
# without attention
class ATONablanet(nn.Module):
    def __init__(self, n_feature, n_linear):
        super(ATONablanet, self).__init__()
        self.linear = torch.nn.Linear(n_feature, n_linear, bias=False)

    def forward(self, anchor, positive, negative):
        embedded_a = self.linear(anchor)
        embedded_p = self.linear(positive)
        embedded_n = self.linear(negative)
        return embedded_a, embedded_p, embedded_n

    def get_lnr(self, x):
        return self.linear(x)


# ---------------------- ATON - ablation -------------------------- #
# without feature embedding module
class ATONabla2net(nn.Module):
    """
    without feature embedding module
    """
    def __init__(self, attn_net):
        super(ATONabla2net, self).__init__()
        self.attn_net = attn_net

    def forward(self, anchor, positive, negative):
        cat = torch.cat([negative, anchor, positive], dim=1)
        attn = self.attn_net(cat)

        embedded_n = negative * attn
        embedded_a = anchor * attn
        embedded_p = positive * attn

        embedded_n_dff = (1 - attn) * negative
        embedded_a_dff = (1 - attn) * anchor
        embedded_p_dff = (1 - attn) * positive
        dis1 = F.pairwise_distance(embedded_n_dff, embedded_a_dff)
        dis2 = F.pairwise_distance(embedded_p_dff, embedded_a_dff)
        dis = torch.abs(dis1 - dis2)

        return embedded_a, embedded_p, embedded_n, attn, dis


# -------------------------- ATON - ablation3 ------------------------------ #
# test the significance of triplet deviation-based loss function

class ATONabla3net(nn.Module):
    def __init__(self, attn_net, clf_net, n_feature, n_linear):
        super(ATONabla3net, self).__init__()
        self.attn_net = attn_net
        self.clf_net = clf_net
        self.linear = torch.nn.Linear(n_feature, n_linear, bias=False)

    def forward(self, x):
        x = self.linear(x)
        attn = self.attn_net(x)
        x = x * attn
        x = self.clf_net(x)
        return x, attn

    def get_lnr(self, x):
        return self.linear(x)


class ClassificationNet(nn.Module):
    def __init__(self, n_feature):
        super(ClassificationNet, self).__init__()
        self.linear = torch.nn.Linear(n_feature, 2)

    def forward(self, x):
        x = self.linear(x)
        return x


class MyLossClf(nn.Module):
    """
    loss function for ablation3
    """
    def __init__(self, alpha1, alpha2, alpha3, margin):
        super(MyLossClf, self).__init__()
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3
        self.criterion_tml = torch.nn.TripletMarginLoss(margin=margin, p=2)
        self.criterion_cel = torch.nn.CrossEntropyLoss()
        return

    def forward(self, embed_anchor, embed_pos, embed_neg, clf_out, batch_y, dis):
        loss_tml = self.criterion_tml(embed_anchor, embed_pos, embed_neg)
        loss_cel = self.criterion_cel(clf_out, batch_y)
        loss_dis = torch.mean(dis)
        loss = self.alpha1 * loss_tml + + self.alpha2 * loss_cel + self.alpha3 * loss_dis
        return loss
