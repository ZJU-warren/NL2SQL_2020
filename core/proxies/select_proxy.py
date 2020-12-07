from core.proxies.select_part.n_cond_proxy import NCondNetProxy
from core.proxies.select_part.cond_operation_proxy import CondOperationNetProxy
from core.proxies.select_part.cond_com_proxy import CondComNetProxy
from core.proxies.select_part.cond_eq_proxy import CondEqNetProxy
from core.proxies.select_part.cond_prefix_proxy import CondPrefixNetProxy
from core.proxies.select_part.cond_suffix_proxy import CondSuffixNetProxy


class SelectProxy:
    def __init__(self, base_net, predict_mode=False, train_data_holder=None, valid_data_holder=None):
        self.mode = predict_mode
        self.n_col_proxy = NColNetProxy(base_net, predict_mode=predict_mode,
                                          train_data_holder=train_data_holder, valid_data_holder=valid_data_holder)

        self.col_prefix_proxy \
            = CondPrefixNetProxy(base_net, predict_mode=predict_mode,
                                 train_data_holder=train_data_holder, valid_data_holder=valid_data_holder)

        self.col_agg_proxy \
            = CondOperationNetProxy(base_net, predict_mode=predict_mode,
                                    train_data_holder=train_data_holder, valid_data_holder=valid_data_holder)

        self.cond_eq_proxy \
            = CondEqNetProxy(base_net, predict_mode=predict_mode,
                             train_data_holder=train_data_holder, valid_data_holder=valid_data_holder)

        self.cond_suffix_proxy \
            = CondSuffixNetProxy(base_net, predict_mode=predict_mode,
                                 train_data_holder=train_data_holder, valid_data_holder=valid_data_holder)

    def run_a_epoch(self):
        self.n_cond_proxy.run_a_epoch()
        self.cond_prefix_proxy.run_a_epoch()
        self.cond_operation_proxy.run_a_epoch()
        self.cond_com_proxy.run_a_epoch()
        self.cond_eq_proxy.run_a_epoch()
        self.cond_suffix_proxy.run_a_epoch()
