from core.models.select_part.module_nets import SelAggNet
from core.proxies.others.proxy import ModuleProxy
from torch.nn import CrossEntropyLoss
from GlobalParameters import cuda_id
import DataLinkSet as DLSet
import json
import numpy as np


class SelectColAggNetProxy(ModuleProxy):
    def __init__(self, base_net, predict_mode=False, train_data_holder=None, valid_data_holder=None, test_data_holder=None):
        super(SelectColAggNetProxy, self).__init__(predict_mode, train_data_holder, valid_data_holder, test_data_holder)
        self._init_env(base_net, SelAggNet, 'Select', 'agg', True)

    def _init_train(self, base_net, target_net, part_name, file_name, tensor=False):
        super()._init_train(base_net, target_net, part_name, file_name, tensor)

        with open(DLSet.main_folder_link % 'Train' + '/Select/X_gt_sup_agg', 'r') as f:
            info = json.load(f)
            self.prefix = np.array(info['prefix'], dtype=np.int32)

        with open(DLSet.main_folder_link % 'Validation' + '/Select/X_gt_sup_agg', 'r') as f:
            info = json.load(f)
            self.valid_prefix = np.array(info['prefix'], dtype=np.int32)

    def _init_test(self, base_net, target_net, part_name, file_name, tensor=False):
        super()._init_test(base_net, target_net, part_name, file_name, tensor)

        # init data
        with open(DLSet.result_folder_link + '/Select/Prefix', 'r') as f:
            info = json.load(f)
            self.X_id = np.array(info['X_id'], dtype=np.int32)
            self.prefix = np.array(info['prefix'], dtype=np.int32)

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
        print("---------------- forward with mode : -------------", self.mode)
        if self.mode == 'Valid':
            y_pd_score = self.target_net(self.train_data_holder, self.X_id[data_index], self.valid_prefix[data_index])
        else:
            y_pd_score = self.target_net(self.train_data_holder, self.X_id[data_index], self.prefix[data_index])
        return self.backward(y_pd_score, data_index, None)

    def backward(self, y_pd, data_index, loss, top=1):
        print(type(data_index))
        gt = self.y_gt[data_index]
        loss = CrossEntropyLoss()(y_pd, gt.cuda(cuda_id))
        return super().backward(y_pd, data_index, loss)

    def predict(self, top=1, keyword=None, target_path=None, extra=None):
        result = super().predict(top, 'agg', '/Select/agg')
