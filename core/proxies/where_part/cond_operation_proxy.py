from core.models.where_part.module_nets import CondOperationNet
from core.proxies.others.proxy import ModuleProxy
from torch.nn import CrossEntropyLoss
from GlobalParameters import cuda_id


class WhereCondOperationNetProxy(ModuleProxy):
    def __init__(self, base_net, predict_mode=False, train_data_holder=None, valid_data_holder=None, test_data_holder=None):
        super(WhereCondOperationNetProxy, self).__init__(predict_mode, train_data_holder, valid_data_holder, test_data_holder)
        self._init_env(base_net, CondOperationNet, 'Where', 'operation', True)

    def predict(self, top=1, keyword=None, target_path=None, extra=None):
        result = super().predict(top, 'operation', '/Where/operation')
        
    def backward(self, y_pd, data_index, loss, top=1):
        gt = self.y_gt[data_index]
        loss = CrossEntropyLoss()(y_pd, gt.cuda(cuda_id))
        return super().backward(y_pd, data_index, loss)


