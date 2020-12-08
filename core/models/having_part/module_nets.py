import torch.nn as nn
from GlobalParameters import cuda_id, max_columns_number
import torch


class CondAggNet(nn.Module):
    MAX_NUM_OF_AGG = 6

    def __init__(self, base_net, hidden=768, gpu=True):
        super(CondAggNet, self).__init__()
        self.base_net = base_net
        self.hidden = hidden
        self.linear = nn.Sequential(nn.Dropout(0.3), nn.Linear(hidden, self.MAX_NUM_OF_AGG))

        if gpu:
            self.cuda(cuda_id)

    def forward(self, data_holder, X_id, sel_cols):
        cls, out, col_att = self.base_net(data_holder, X_id)
        score = torch.empty((len(sel_cols), 768)).cuda(cuda_id)
        for i in range(len(sel_cols)):
            score[i] = out[i, sel_cols[i], :].squeeze()
        score = self.linear(score)
        return score
