from core.models.where_part.module_nets import CondComNet
from core.proxies.others.proxy import ModuleProxy
from torch.nn import CrossEntropyLoss
from GlobalParameters import cuda_id

import DataLinkSet as DLSet
from tools.metrics import acc
import random
import json
import numpy as np


class WhereCondComNetProxy(ModuleProxy):
    def __init__(self, base_net, predict_mode=False, train_data_holder=None, valid_data_holder=None, test_data_holder=None):
        super(WhereCondComNetProxy, self).__init__(predict_mode, train_data_holder, valid_data_holder, test_data_holder)
        self._init_env(base_net, CondComNet, 'Where', 'com', True)

    def _init_train(self, base_net, target_net, part_name, file_name, tensor=False):
        super()._init_train(base_net, target_net, part_name, file_name, tensor)

        with open(DLSet.main_folder_link % 'Train' + '/Where/X_gt_sup_com', 'r') as f:
            info = json.load(f)
            self.prefix = np.array(info['prefix'], dtype=np.int32)

        with open(DLSet.main_folder_link % 'Validation' + '/Where/X_gt_sup_com', 'r') as f:
            info = json.load(f)
            self.valid_prefix = np.array(info['prefix'], dtype=np.int32)

    def _init_test(self, base_net, target_net, part_name, file_name, tensor=False):
        super()._init_test(base_net, target_net, part_name, file_name, tensor)

        # init data
        with open(DLSet.result_folder_link + '/Where/prefix', 'r') as f:
            info = json.load(f)
            self.X_id = np.array(info['X_id'], dtype=np.int32)
            self.prefix = info['prefix']

            X_id = []
            prefix = []

            num = len(self.X_id)
            for i in range(num):
                for each in self.prefix[i]:
                    X_id.append(self.X_id[i])
                    prefix.append(each)

            self.prefix = np.array(prefix, dtype=np.int32)
            self.X_id = np.array(X_id, dtype=np.int32)

        # init data
        self.total = self.X_id.shape[0]

    def forward(self, data_index):
        y_pd_score = self.target_net(self.train_data_holder, self.X_id[data_index], self.prefix[data_index])
        return self.backward(y_pd_score, data_index, None)

    def predict(self, top=1, keyword=None, target_path=None, extra=None):
        result = super().predict(top, 'com', '/Where/com', extra=self.prefix)

    def backward(self, y_pd, data_index, loss, top=1):
        gt = self.y_gt[data_index]
        loss = CrossEntropyLoss()(y_pd, gt.cuda(cuda_id))

        self.avg_loss = (self.avg_loss * self.step + loss.data.cpu().numpy()) / (self.step + 1)

        self.step += 1
        self.loss = loss
        self.loss.backward()

        acc_value_valid = -1

        if self.step % 10 == 0:
            print('-- loss_cpu', self.loss.data.cpu().numpy())
            self.optimizer.step()
            self.optimizer.zero_grad()

            # calculate acc
            # @train
            gt = self.y_gt[data_index]
            acc_value = acc(y_pd.data.cpu().numpy(), gt)
            print('%s -- acc@train' % self.__class__.__name__, acc_value)

            # @validation
            total_valid = len(self.valid_y_gt)
            data_index = random.sample([i for i in range(total_valid)], 15)
            # data_index = [i for i in range(total_valid)]
            gt = self.valid_y_gt[data_index]
            y_pd_valid = self.target_net(self.train_data_holder, self.X_id[data_index], self.valid_prefix[data_index])
            acc_value_valid = acc(y_pd_valid.data.cpu().numpy(), gt)
            print('%s -- acc@valid' % self.__class__.__name__, acc_value_valid)
            self.step = 0

        return acc_value_valid