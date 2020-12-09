from core.models.select_part.module_nets import SelSuffixColNet
from core.proxies.others.proxy import ModuleProxy
import torch
from GlobalParameters import cuda_id, max_columns_number
import DataLinkSet as DLSet
import json
import numpy as np
from torch.nn import CrossEntropyLoss


class SelectColSuffixNetProxy(ModuleProxy):

    def _init_test(self, base_net, target_net, part_name, file_name, tensor=False):
        super()._init_test(base_net, target_net, part_name, file_name, tensor)

        with open(DLSet.result_folder_link + '/Select/K', 'r') as f:
            info = json.load(f)
            self.X_id = info['X_id']
            prefix_N = info['K']

            X_id = []
            num = len(self.X_id)
            for i in range(num):
                if prefix_N[i] != 0:
                    for k in range(prefix_N[i]):
                        X_id.append(self.X_id[i])
            self.X_id = np.array(X_id, dtype=np.int32)

        # init data
        self.total = self.X_id.shape[0]

    def __init__(self, base_net, predict_mode=False, train_data_holder=None, valid_data_holder=None, test_data_holder=None, thres=0.90):
        super(SelectColSuffixNetProxy, self).__init__(predict_mode, train_data_holder, valid_data_holder, test_data_holder, thres=thres)
        self._init_env(base_net, SelSuffixColNet, 'Select', 'suffix', True)

    def backward(self, y_pd, data_index, loss, top=1):
        gt = self.y_gt[data_index]
        loss = CrossEntropyLoss()(y_pd, gt.cuda(cuda_id))
        return super().backward(y_pd, data_index, loss)

    def predict(self, top=1, keyword=None, target_path=None, extra=None):
        result = super().predict(top, 'suffix', '/Select/suffix')
