from core.models.from_part.module_nets import CondSuffixNet
from core.proxies.others.proxy import ModuleProxy
import torch
from GlobalParameters import cuda_id, max_columns_number
import DataLinkSet as DLSet
from tools.metrics import acc
import random
import json
import numpy as np


class FromCondSuffixNetProxy(ModuleProxy):
    def __init__(self, base_net, predict_mode=False, train_data_holder=None,
                 valid_data_holder=None, test_data_holder=None):
        super(FromCondSuffixNetProxy, self).__init__(predict_mode, train_data_holder, valid_data_holder,
                                                     test_data_holder)
        self._init_env(base_net, CondSuffixNet, 'From', 'suffix')

    def predict(self, top=1, keyword=None, target_path=None, extra=None):
        result = super().predict(top, 'suffix', '/From/suffix', extra=self.prefix)

    def _init_train(self, base_net, target_net, part_name, file_name, tensor=False):
        super()._init_train(base_net, target_net, part_name, file_name, tensor)
        # init data
        self.sel_col_for_loss = torch.zeros((self.y_gt.shape[0], max_columns_number)).cuda(cuda_id)
        for i in range(self.y_gt.shape[0]):
            self.sel_col_for_loss[i, self.y_gt[i]] = 1
        self.header_mask = torch.zeros((self.y_gt.shape[0], max_columns_number)).cuda(cuda_id)

        for i in range(self.y_gt.shape[0]):
            col_num = self.train_data_holder.col_num(self.X_id[i])
            self.header_mask[i, :col_num] = 1

        # init data
        self.sel_col_for_loss = torch.zeros((self.y_gt.shape[0], max_columns_number)).cuda(cuda_id)
        for i in range(self.y_gt.shape[0]):
            self.sel_col_for_loss[i, self.y_gt[i]] = 1
        self.header_mask = torch.zeros((self.y_gt.shape[0], max_columns_number)).cuda(cuda_id)

        for i in range(self.y_gt.shape[0]):
            col_num = self.train_data_holder.col_num(i)
            self.header_mask[i, :col_num] = 1

        with open(DLSet.main_folder_link % 'Train' + '/From/X_gt_sup_suffix', 'r') as f:
            info = json.load(f)
            self.prefix = np.array(info['prefix'], dtype=np.int32)

        with open(DLSet.main_folder_link % 'Validation' + '/From/X_gt_sup_suffix', 'r') as f:
            info = json.load(f)
            self.valid_prefix = np.array(info['prefix'], dtype=np.int32)

    def _init_test(self, base_net, target_net, part_name, file_name, tensor=False):
        super()._init_test(base_net, target_net, part_name, file_name, tensor)

        # init data
        with open(DLSet.result_folder_link + '/From/prefix', 'r') as f:
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

        self.header_mask = torch.zeros((self.X_id.shape[0], max_columns_number)).cuda(cuda_id)
        for i in range(self.X_id.shape[0]):
            col_num = self.test_data_holder.col_num(self.X_id[i])
            self.header_mask[i, :col_num] = 1

        # init data
        self.total = self.X_id.shape[0]

    def forward(self, data_index):
        y_pd_score = self.target_net(self.train_data_holder, self.X_id[data_index], self.prefix[data_index])
        return self.backward(y_pd_score, data_index, None)

    def backward(self, y_pd, data_index, loss, top=1):
        sel_col_label = self.sel_col_for_loss[data_index]

        col_raw_loss = 3 * sel_col_label * torch.log(y_pd + 1e-10) \
                       + (1 - sel_col_label) * torch.log(1 - y_pd + 1e-10)
        col_mask_loss = col_raw_loss * self.header_mask[data_index]
        loss = -torch.sum(col_mask_loss) / torch.sum(self.header_mask[data_index])

        self.avg_loss = (self.avg_loss * self.step + loss.data.cpu().numpy()) / (self.step + 1)

        self.step += 1
        self.loss = loss
        self.loss.backward()

        acc_value_valid = -1

        if self.step % 1 == 0:
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
            data_index = random.sample([i for i in range(total_valid)], 1)
            # data_index = [i for i in range(total_valid)]
            gt = self.valid_y_gt[data_index]
            y_pd_valid = self.target_net(self.train_data_holder, self.X_id[data_index],
                                         self.valid_prefix[data_index])
            acc_value_valid = acc(y_pd_valid.data.cpu().numpy(), gt)
            print('%s -- acc@valid' % self.__class__.__name__, acc_value_valid)
            self.step = 0

        return acc_value_valid
