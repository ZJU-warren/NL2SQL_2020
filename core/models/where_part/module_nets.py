import torch.nn as nn
from GlobalParameters import cuda_id, max_columns_number
import torch


class CondOperationNet(nn.Module):
    MAX_NUM_OF_OPERATION = 2

    def __init__(self, base_net, hidden=768, gpu=True):
        super(CondOperationNet, self).__init__()
        self.base_net = base_net
        self.hidden = hidden
        self.linear = nn.Sequential(nn.Dropout(0.3), nn.Linear(hidden, self.MAX_NUM_OF_OPERATION))

        if gpu:
            self.cuda(cuda_id)

    def forward(self, data_holder, X_id):
        cls, out, col_att = self.base_net(data_holder, X_id)
        score = self.linear(cls)
        return score


class CondComNet(nn.Module):
    MAX_NUM_OF_COM = 5

    def __init__(self, base_net, hidden=768, gpu=True):
        super(CondComNet, self).__init__()
        self.base_net = base_net
        self.hidden = hidden
        self.linear = nn.Sequential(nn.Dropout(0.3), nn.Linear(hidden, self.MAX_NUM_OF_COM))

        if gpu:
            self.cuda(cuda_id)

    def forward(self, data_holder, X_id, sel_cols):
        cls, out, col_att = self.base_net(data_holder, X_id)
        score = torch.empty((len(sel_cols), 768)).cuda()
        for i in range(len(sel_cols)):
            score[i] = out[i, sel_cols[i], :].squeeze()
        score = self.out(score)
        return score


class CondEqNet(nn.Module):
    MAX_NUM_OF_EQ = 11

    def __init__(self, base_net, hidden=768, gpu=True):
        super(CondEqNet, self).__init__()
        self.base_net = base_net
        self.hidden = hidden
        self.linear = nn.Sequential(nn.Dropout(0.3), nn.Linear(hidden, self.MAX_NUM_OF_EQ))

        if gpu:
            self.cuda(cuda_id)

    def forward(self, data_holder, X_id, sel_cols):
        cls, out, col_att = self.base_net(data_holder, X_id)
        score = torch.empty((len(sel_cols), 768)).cuda()
        for i in range(len(sel_cols)):
            score[i] = out[i, sel_cols[i], :].squeeze()
        score = self.out(score)
        return score
