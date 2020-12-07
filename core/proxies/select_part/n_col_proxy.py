from core.models.select_part.module_nets import SelNumNet
from core.proxies.others.proxy import ModuleProxy
from torch.nn import CrossEntropyLoss
from GlobalParameters import cuda_id


class NColNetProxy(ModuleProxy):
    def __init__(self, base_net, predict_mode=False, train_data_holder=None, valid_data_holder=None):
        super(NColNetProxy, self).__init__(predict_mode, train_data_holder, valid_data_holder)
        self._init_env(base_net, SelNumNet, 'Select', 'K')

    def backward(self, y_pd, data_index, loss, top=1):
        gt = self.y_gt[data_index]
        loss = CrossEntropyLoss()(y_pd, gt.cuda(cuda_id))
        super().backward(y_pd, data_index, loss)
