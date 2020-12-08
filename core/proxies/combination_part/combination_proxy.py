from core.models.combination_part.module_nets import CombinationNet
from core.proxies.others.proxy import ModuleProxy
from torch.nn import CrossEntropyLoss
from GlobalParameters import cuda_id


class CombinationNetProxy(ModuleProxy):
    def __init__(self, base_net, predict_mode=False, train_data_holder=None, valid_data_holder=None, test_data_holder=None):
        super(CombinationNetProxy, self).__init__(predict_mode, train_data_holder, valid_data_holder, test_data_holder)
        self._init_env(base_net, CombinationNet, 'Combination', 'comb', True)

    def backward(self, y_pd, data_index, loss, top=1):
        gt = self.y_gt[data_index]
        loss = CrossEntropyLoss()(y_pd, gt.cuda(cuda_id))
        return super().backward(y_pd, data_index, loss)

    def predict(self, top=1, keyword=None, target_path=None):
        result = super().predict(top, 'comb', '/Combination/comb')