import torch.nn as nn
from GlobalParameters import cuda_id, max_columns_number, max_agg_number, max_op_number
import torch


class SelNumNet(nn.Module):
    MAX_COL = 5

    def __init__(self, base_net, hidden=768, gpu=True):
        super(SelNumNet, self).__init__()
        self.base_net = base_net
        self.hidden = hidden
        self.linear = nn.Sequential(nn.Dropout(0.3), nn.Linear(hidden, self.MAX_COL))

        if gpu:
            self.cuda(cuda_id)

    def forward(self, data_holder, X_id):
        cls, out, col_att = self.base_net(data_holder, X_id)
        score = self.linear(cls)
        return score


class SelColNet(nn.Module):
    def __init__(self, base_net, hidden=768, gpu=True):
        super(SelColNet, self).__init__()
        self.base_net = base_net
        self.C = max_columns_number

        self.out = nn.Sequential(nn.LayerNorm(hidden), nn.Dropout(0.2), nn.Linear(hidden, 1), nn.Sigmoid())

        if gpu:
            self.cuda(cuda_id)

    def forward(self,data_holder, X_id):
        cls, out, col_att = self.base_net(data_holder, X_id)
        score = self.out(out).squeeze(-1)
        return score


class SelAggNet(nn.Module):
    def __init__(self, base_net, hidden=768, gpu=True):
        super(SelAggNet, self).__init__()
        self.base_net = base_net
        self.C = max_agg_number
        self.out = nn.Linear(hidden, self.C)

        if gpu:
            self.cuda(cuda_id)

    def forward(self,data_holder, X_id, sel_cols):
        b = len(inp)
        col_len = [len(col) for col in sel_cols]
        t = sum(col_len)
        if self.gpu:
            score = torch.zeros(t, self.C).cuda(cuda_id)
        else:
            score = torch.zeros(t, self.C)
        st = 0
        for i in range(b):
            ed = st+col_len[i]
            score[st:ed, :] = self.out(inp[i, sel_cols[i]])
            st = ed
        return score


class SelOpNet(nn.Module):
    def __init__(self, base_net, hidden=768, gpu=True):
        super(SelOpNet, self).__init__()
        self.base_net = base_net
        self.C = max_op_number
        self.out = nn.Linear(hidden, self.C)
        self.linear = nn.Sequential(nn.Dropout(0.3), nn.Linear(hidden, self.C))
        self.has_op = nn.Sequential(nn.Dropout(0.3), nn.Linear(hidden, 1), nn.Sigmoid())

        if gpu:
            self.cuda(cuda_id)

    def forward(self, cls, data_holder, X_id, sel_cols):
        cls, out, col_att = self.base_net(data_holder, X_id)

        return self.linear(cls), self.has_op(cls).squeeze()


class SelSuffixColNet(nn.Module):
    def __init__(self, base_net, hidden=768, gpu=True):
        super(SelSuffixColNet, self).__init__()
        self.base_net = base_net
        self.C = max_columns_number

        self.out = nn.Sequential(nn.LayerNorm(hidden), nn.Dropout(0.2), nn.Linear(hidden, 1))

        if gpu:
            self.cuda(cuda_id)

    def forward(self, data_holder, X_id):
        cls, out, col_att = self.base_net(data_holder, X_id)
        score = self.out(out).squeeze(-1)
        return score
