from core.models.from_part.module_nets import NCondNet
import DataLinkSet as DLSet
import json
import numpy as np
from core.proxies.others.proxy import ModuleProxy
from torch.nn import CrossEntropyLoss, MSELoss, BCELoss
from GlobalParameters import learning_rate
import torch.optim as optim
import torch
from GlobalParameters import cuda_id


def acc(pred, gt):
    total = pred.shape[0]
    shot = 0
    for _ in range(total):
        n_pred = np.argmax(pred[_])
        shot += 1 if n_pred == gt[_] else 0
    return shot / total


class NCondNetProxy(ModuleProxy):
    def __init_train(self, base_net):
        # init model
        self.N_net = NCondNet(base_net)

        # load data
        with open(DLSet.main_folder_link % 'Train' + '/From/X_gt_sup_N', 'r') as f:
            self.X_id = np.array(json.load(f)['X_id'], dtype=np.int32)

        with open(DLSet.main_folder_link % 'Train' + '/From/y_gt_N', 'r') as f:
            self.y_gt_N = torch.Tensor(json.load(f)['N']).long()

        return self.y_gt_N.shape[0]

    def __init__(self, base_net, predict_mode=False):
        total = self.__init_train(base_net)
        super(NCondNetProxy, self).__init__(total, predict_mode)

        self.loss = 0
        self.step = 0
        self.acc = 0
        self.optimizer = optim.Adam(self.N_net.parameters(), lr=learning_rate)

    def forward(self, data_index):
        y_pd_N_score = self.N_net(self.X_id[data_index])

        if self.mode is False:
            self.backward(y_pd_N_score, data_index)

    def backward(self, y_pd_N_score, data_index):
        self.step += 1
        gt = self.y_gt_N[data_index]
        self.loss = CrossEntropyLoss()(y_pd_N_score, gt.cuda(cuda_id))
        self.loss.backward()

        if self.step % 10 == 0:
            print('loss_cpu', self.loss.data.cpu().numpy())

            self.optimizer.step()
            self.optimizer.zero_grad()
            self.step = 0

            acc_cpu_test = acc(y_pd_N_score.data.cpu().numpy(), gt)
            print('acc_cpu_test', acc_cpu_test)


class CondPrefixNetProxy(ModuleProxy):
    def __init_train(self, base_net):
        # init model
        self.N_net = NCondNet(base_net)

        # load data
        with open(DLSet.main_folder_link % 'Train' + '/From/X_gt_sup_prefix', 'r') as f:
            self.X_id = np.array(json.load(f)['X_id'], dtype=np.int32)

        with open(DLSet.main_folder_link % 'Train' + '/From/y_gt_suffix', 'r') as f:
            self.y_gt_prefix = torch.Tensor(json.load(f)['prefix']).long()

        return self.y_gt_prefix.shape[0]

    def __init__(self, base_net, predict_mode=False):
        total = self.__init_train(base_net)
        super(CondPrefixNetProxy, self).__init__(total, predict_mode)

        self.loss = 0
        self.step = 0
        self.acc = 0
        self.optimizer = optim.Adam(self.N_net.parameters(), lr=learning_rate)

    def forward(self, data_index):
        y_pd_N_score = self.N_net(self.X_id[data_index])

        if self.mode is False:
            self.backward(y_pd_N_score, data_index)

    def backward(self, y_pd_N_score, data_index):
        self.step += 1
        gt = self.y_gt_prefix[data_index]
        self.loss = CrossEntropyLoss()(y_pd_N_score, gt.cuda(cuda_id))
        self.loss.backward()

        if self.step % 10 == 0:
            print('loss_cpu', self.loss.data.cpu().numpy())

            self.optimizer.step()
            self.optimizer.zero_grad()
            self.step = 0

            acc_cpu_test = acc(y_pd_N_score.data.cpu().numpy(), gt)
            print('acc_cpu_test', acc_cpu_test)
