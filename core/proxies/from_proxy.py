from core.proxies.from_part.n_cond_proxy import NCondNetProxy
from core.proxies.from_part.j_cond_proxy import JCondNetProxy
from core.proxies.from_part.cond_prefix_proxy import CondPrefixNetProxy
from core.proxies.from_part.cond_suffix_proxy import CondSuffixNetProxy


class FromProxy:
    def __init__(self, base_net, predict_mode=False, train_data_holder=None, valid_data_holder=None):
        self.mode = predict_mode
        self.n_cond_proxy = NCondNetProxy(base_net, predict_mode=predict_mode,
                                          train_data_holder=train_data_holder, valid_data_holder=valid_data_holder)

        self.cond_prefix_proxy \
            = CondPrefixNetProxy(base_net, predict_mode=predict_mode,
                                 train_data_holder=train_data_holder, valid_data_holder=valid_data_holder)

        self.cond_suffix_proxy \
            = CondSuffixNetProxy(base_net, predict_mode=predict_mode,
                                 train_data_holder=train_data_holder, valid_data_holder=valid_data_holder)

        self.j_cond_proxy = JCondNetProxy(base_net, predict_mode=predict_mode,
                                          train_data_holder=train_data_holder, valid_data_holder=valid_data_holder)

    def run_a_epoch(self):
        self.n_cond_proxy.run_a_epoch()
        self.cond_prefix_proxy.run_a_epoch()
        self.cond_suffix_proxy.run_a_epoch()
        self.j_cond_proxy.run_a_epoch()
