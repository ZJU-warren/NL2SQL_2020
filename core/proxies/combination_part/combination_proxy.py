from core.models.combination_part.module_nets import CombinationNet
from core.proxies.others.proxy import ModuleProxy
from torch.nn import CrossEntropyLoss
from GlobalParameters import cuda_id


class CombinationNetProxy(ModuleProxy):
    def _init_train(self, base_net, target_net, part_name, file_name, tensor=False):
        super()._init_train(base_net, target_net, part_name, file_name, tensor=True)

    def __init__(self, base_net, predict_mode=False, train_data_holder=None, valid_data_holder=None):
        super(CombinationNetProxy, self).__init__(predict_mode, train_data_holder, valid_data_holder)
        self._init_train(base_net, CombinationNet, 'Combination', 'comb')

    def backward(self, y_pd, data_index, loss, top=1):
        gt = self.y_gt[data_index]
        loss = CrossEntropyLoss()(y_pd, gt.cuda(cuda_id))
        super().backward(y_pd, data_index, loss)
