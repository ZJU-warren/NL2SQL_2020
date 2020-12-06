import torch.nn as nn
from GlobalParameters import cuda_id, max_columns_number


class NCondNet(nn.Module):
    MAX_NUM_OF_N = 3

    def __init__(self, base_net, hidden=768, gpu=True):
        super(NCondNet, self).__init__()
        self.base_net = base_net
        self.hidden = hidden
        self.linear = nn.Sequential(nn.Dropout(0.3), nn.Linear(hidden, self.MAX_NUM_OF_N))

        if gpu:
            self.cuda(cuda_id)

    def forward(self, X_id):
        cls, out, col_att = self.base_net(X_id)
        score = self.linear(cls)
        return score


class CondPrefixModule(nn.Module):
    MAX_NUM_OF_COL = max_columns_number

    def __init__(self, base_net, hidden=768, gpu=True):
        super(CondPrefixModule, self).__init__()
        self.base_net = base_net
        self.hidden = hidden
        self.out = nn.Sequential(nn.LayerNorm(hidden), nn.Dropout(0.2), nn.Linear(hidden, 1), nn.Sigmoid())

        if gpu:
            self.cuda(cuda_id)

    def forward(self, X_id):
        cls, out, col_att = self.base_net(X_id)
        score = self.out(out).squeeze(-1)
        return score
