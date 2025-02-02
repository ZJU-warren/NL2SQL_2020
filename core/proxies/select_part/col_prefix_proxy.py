from core.models.from_part.module_nets import CondPrefixNet
from core.proxies.others.proxy import ModuleProxy
import torch
from GlobalParameters import max_columns_number, cuda_id
import DataLinkSet as DLSet
import json
import numpy as np


class SelectColPrefixNetProxy(ModuleProxy):
    def _init_train(self, base_net, target_net, part_name, file_name, tensor=False):
        super()._init_train(base_net, target_net, part_name, file_name, tensor)

        with open(DLSet.main_folder_link % 'Train' + '/Select/X_gt_sup_prefix', 'r') as f:
            info = json.load(f)
            self.prefix_N = np.array(info['K'], dtype=np.int32)

        with open(DLSet.main_folder_link % 'Validation' + '/Select/X_gt_sup_prefix', 'r') as f:
            info = json.load(f)
            self.valid_prefix_N = np.array(info['K'], dtype=np.int32)

        # init data
        self.sel_col_for_loss = torch.zeros((self.y_gt.shape[0], max_columns_number)).cuda(cuda_id)
        for i in range(self.y_gt.shape[0]):
            self.sel_col_for_loss[i, self.y_gt[i]] = 1

        self.header_mask = torch.zeros((self.y_gt.shape[0], max_columns_number)).cuda(cuda_id)
        for i in range(self.y_gt.shape[0]):
            col_num = self.train_data_holder.col_num(self.X_id[i])
            self.header_mask[i, :col_num] = 1

    def _init_test(self, base_net, target_net, part_name, file_name, tensor=False):
        super()._init_test(base_net, target_net, part_name, file_name, tensor)

        # init data
        with open(DLSet.result_folder_link + '/Select/K', 'r') as f:
            info = json.load(f)
            self.X_id = info['X_id']
            self.prefix_N = info['K']

            X_id = []
            prefix = []
            print('before', self.prefix_N)

            num = len(self.X_id)
            for i in range(num):
                if self.prefix_N[i] != 0:
                    X_id.append(self.X_id[i])
                    prefix.append(self.prefix_N[i])

            self.prefix_N = np.array(prefix, dtype=np.int32)
            self.X_id = np.array(X_id, dtype=np.int32)
            print('after', self.prefix_N)

        # init data
        self.total = self.X_id.shape[0]

        # init data
        self.header_mask = torch.zeros((self.X_id.shape[0], max_columns_number)).cuda(cuda_id)
        for i in range(self.X_id.shape[0]):
            col_num = self.test_data_holder.col_num(self.X_id[i])
            self.header_mask[i, :col_num] = 1

    def __init__(self, base_net, predict_mode=False, train_data_holder=None, valid_data_holder=None, test_data_holder=None):
        super(SelectColPrefixNetProxy, self).__init__(predict_mode, train_data_holder, valid_data_holder, test_data_holder, thres=0.85)

        self._init_env(base_net, CondPrefixNet, 'Select', 'prefix')

    def backward(self, y_pd, data_index, loss, top=None):
        sel_col_label = self.sel_col_for_loss[data_index]
        col_raw_loss = 3 * sel_col_label * torch.log(y_pd + 1e-10) \
                       + (1 - sel_col_label) * torch.log(1 - y_pd + 1e-10)
        col_mask_loss = col_raw_loss * self.header_mask[data_index]
        loss = -torch.sum(col_mask_loss) / torch.sum(self.header_mask[data_index])
        return super().backward(y_pd, data_index, loss, top=(self.prefix_N, self.valid_prefix_N))

    def predict(self, top=1, keyword=None, target_path=None, extra=None):
        result = super().predict(self.prefix_N, 'prefix', '/Select/prefix')
