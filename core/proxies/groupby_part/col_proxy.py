from core.models.from_part.module_nets import CondSuffixNet
from core.proxies.others.proxy import ModuleProxy
import torch
from GlobalParameters import cuda_id, max_columns_number


class GroupByColNetProxy(ModuleProxy):
    def _init_train(self, base_net, target_net, part_name, file_name, tensor=False):
        super()._init_train(base_net, target_net, part_name, file_name)

        # init data
        self.sel_col_for_loss = torch.zeros((self.y_gt.shape[0], max_columns_number)).cuda(cuda_id)
        for i in range(self.y_gt.shape[0]):
            self.sel_col_for_loss[i, self.y_gt[i]] = 1
        self.header_mask = torch.zeros((self.y_gt.shape[0], max_columns_number)).cuda(cuda_id)

        for i in range(self.y_gt.shape[0]):
            col_num = self.train_data_holder.col_num(self.X_id[i])
            self.header_mask[i, :col_num] = 1

    def __init__(self, base_net, predict_mode=False, train_data_holder=None, valid_data_holder=None, test_data_holder=None):
        super(GroupByColNetProxy, self).__init__(predict_mode, train_data_holder, valid_data_holder, test_data_holder)
        self._init_env(base_net, CondSuffixNet, 'GroupBy', 'col')

    def backward(self, y_pd, data_index, loss, top=1):
        sel_col_label = self.sel_col_for_loss[data_index]

        col_raw_loss = 3 * sel_col_label * torch.log(y_pd + 1e-10) \
                       + (1 - sel_col_label) * torch.log(1 - y_pd + 1e-10)
        col_mask_loss = col_raw_loss * self.header_mask[data_index]
        loss = -torch.sum(col_mask_loss) / torch.sum(self.header_mask[data_index])
        return super().backward(y_pd, data_index, loss)

    def predict(self, top=1, keyword=None, target_path=None):
        result = super().predict(top, 'col', '/GroupBy/col')