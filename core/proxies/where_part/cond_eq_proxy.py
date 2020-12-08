from core.models.where_part.module_nets import CondEqNet
from core.proxies.others.proxy import ModuleProxy
from torch.nn import CrossEntropyLoss
from GlobalParameters import cuda_id


class WhereCondEqNetProxy(ModuleProxy):
    def __init__(self, base_net, predict_mode=False, train_data_holder=None, valid_data_holder=None, test_data_holder=None):
        super(WhereCondEqNetProxy, self).__init__(predict_mode, train_data_holder, valid_data_holder, test_data_holder)
        self._init_env(base_net, CondEqNet, 'Where', 'eq', True)

    def backward(self, y_pd, data_index, loss, top=1):
        gt = self.y_gt[data_index]
        loss = CrossEntropyLoss()(y_pd, gt.cuda(cuda_id))
        return super().backward(y_pd, data_index, loss)

    def predict(self, top=1, keyword=None, target_path=None):
        result = super().predict(top, 'eq', '/Where/eq')
