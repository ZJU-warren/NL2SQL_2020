from core.models.select_part.module_nets import SelSuffixColNet
from core.proxies.others.proxy import ModuleProxy
import torch
from GlobalParameters import cuda_id, max_columns_number
from torch.nn import CrossEntropyLoss


class GroupByColNetProxy(ModuleProxy):
    def __init__(self, base_net, predict_mode=False, train_data_holder=None,
                 valid_data_holder=None, test_data_holder=None, thres=0.9):
        super(GroupByColNetProxy, self).__init__(predict_mode, train_data_holder, valid_data_holder, test_data_holder, thres=thres)
        self._init_env(base_net, SelSuffixColNet, 'GroupBy', 'col', True)

    def backward(self, y_pd, data_index, loss, top=1):
        gt = self.y_gt[data_index]
        loss = CrossEntropyLoss()(y_pd, gt.cuda(cuda_id))
        return super().backward(y_pd, data_index, loss)

    def predict(self, top=1, keyword=None, target_path=None, extra=None):
        result = super().predict(top, 'col', '/GroupBy/col')