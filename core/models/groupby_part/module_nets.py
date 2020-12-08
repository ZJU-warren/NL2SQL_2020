import torch.nn as nn
from GlobalParameters import cuda_id, max_columns_number


class ColOneNet(nn.Module):
    MAX_NUM_OF_COL = 44

    def __init__(self, base_net, hidden=768, gpu=True):
        super(ColOneNet, self).__init__()
        self.base_net = base_net
        self.hidden = hidden
        self.linear = nn.Sequential(nn.Dropout(0.3), nn.Linear(hidden, self.MAX_NUM_OF_COL))

        if gpu:
            self.cuda(cuda_id)

    def forward(self, data_holder, X_id):
        cls, out, col_att = self.base_net(data_holder, X_id)
        score = self.linear(cls)
        return score


class NeedNet(nn.Module):
    MAX_NUM_OF_NEED = 2

    def __init__(self, base_net, hidden=768, gpu=True):
        super(NeedNet, self).__init__()
        self.base_net = base_net
        self.hidden = hidden
        self.linear = nn.Sequential(nn.Dropout(0.3), nn.Linear(hidden, self.MAX_NUM_OF_NEED))

        if gpu:
            self.cuda(cuda_id)

    def forward(self, data_holder, X_id):
        cls, out, col_att = self.base_net(data_holder, X_id)
        score = self.linear(cls)
        return score
