from core.models.select_part.module_nets import SelOpNet
from core.proxies.others.proxy import ModuleProxy
from torch.nn import CrossEntropyLoss
from GlobalParameters import cuda_id

import DataLinkSet as DLSet
import json
import numpy as np
from tools.metrics import acc
import random


class SelectColComNetProxy(ModuleProxy):
    def _init_train(self, base_net, target_net, part_name, file_name, tensor=False):
        super()._init_train(base_net, target_net, part_name, file_name, tensor=True)

    def __init__(self, base_net, predict_mode=False, train_data_holder=None,
                 valid_data_holder=None, test_data_holder=None):
        super(SelectColComNetProxy, self).__init__(predict_mode, train_data_holder, valid_data_holder, test_data_holder)
        self._init_env(base_net, SelOpNet, 'Select', 'com', True)

    # init data
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

    def predict(self, top=1, keyword=None, target_path=None, extra=None):
        result = super().predict(top, 'com', '/Select/com')

    def backward(self, y_pd, data_index, loss, top=1):
        gt = self.y_gt[data_index]
        loss = CrossEntropyLoss()(y_pd, gt.cuda(cuda_id))
        return super().backward(y_pd, data_index, loss)


