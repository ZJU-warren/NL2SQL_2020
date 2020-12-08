import torch.nn as nn
from GlobalParameters import cuda_id, max_columns_number
import torch


class NCondNet(nn.Module):
    MAX_NUM_OF_N = 3

    def __init__(self, base_net, hidden=768, gpu=True):
        super(NCondNet, self).__init__()
        self.base_net = base_net
        self.hidden = hidden
        self.linear = nn.Sequential(nn.Dropout(0.3), nn.Linear(hidden, self.MAX_NUM_OF_N))

        if gpu:
            self.cuda(cuda_id)

    def forward(self, data_holder, X_id):
        cls, out, col_att = self.base_net(data_holder, X_id)
        score = self.linear(cls)
        return score


class CondPrefixNet(nn.Module):
    MAX_NUM_OF_COL = max_columns_number

    def __init__(self, base_net, hidden=768, gpu=True):
        super(CondPrefixNet, self).__init__()
        self.base_net = base_net
        self.hidden = hidden
        self.out = nn.Sequential(nn.LayerNorm(hidden), nn.Dropout(0.2), nn.Linear(hidden, 1), nn.Sigmoid())
        # self.out = nn.Sequential(nn.Dropout(0.3), nn.Linear(hidden, self.MAX_NUM_OF_COL), nn.Sigmoid())

        if gpu:
            self.cuda(cuda_id)

    def forward(self, data_holder, X_id):
        cls, out, col_att = self.base_net(data_holder, X_id)
        score = self.out(out).squeeze(-1)
        # score = self.out(cls).squeeze(-1)
        return score


class CondSuffixNet(nn.Module):
    MAX_NUM_OF_COL = max_columns_number

    def __init__(self, base_net, hidden=768, gpu=True):
        super(CondSuffixNet, self).__init__()
        self.base_net = base_net
        self.hidden = hidden
        self.out = nn.Linear(hidden, self.MAX_NUM_OF_COL)
        # self.out = nn.Sequential(nn.Dropout(0.3), nn.Linear(hidden, self.MAX_NUM_OF_COL), nn.Sigmoid())
        if gpu:
            self.cuda(cuda_id)

    def forward(self, data_holder, X_id, sel_cols):
        cls, out, col_att = self.base_net(data_holder, X_id)
        score = torch.empty((len(sel_cols), 768)).cuda(cuda_id)
        for i in range(len(sel_cols)):
            score[i] = out[i, sel_cols[i], :].squeeze()
        score = self.out(score)
        return score


class JCondNet(nn.Module):
    MAX_NUM_OF_J = 4

    def __init__(self, base_net, hidden=768, gpu=True):
        super(JCondNet, self).__init__()
        self.base_net = base_net
        self.hidden = hidden
        self.linear = nn.Sequential(nn.Dropout(0.3), nn.Linear(hidden, self.MAX_NUM_OF_J))

        if gpu:
            self.cuda(cuda_id)

    def forward(self, data_holder, X_id):
        cls, out, col_att = self.base_net(data_holder, X_id)
        score = self.linear(cls)
        return score
