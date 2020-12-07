from core.proxies.select_part.n_col_proxy import NColNetProxy
# from core.proxies.select_part.col_prefix_proxy import ColPrefixNetProxy
# from core.proxies.select_part.col_agg_proxy import ColAggNetProxy
# from core.proxies.select_part.col_com_proxy import ColComNetProxy
# from core.proxies.select_part.col_suffix_proxy import ColSuffixNetProxy


class SelectProxy:
    def __init__(self, base_net, predict_mode=False, train_data_holder=None, valid_data_holder=None):
        self.mode = predict_mode
        self.n_col_proxy = NColNetProxy(base_net, predict_mode=predict_mode,
                                        train_data_holder=train_data_holder, valid_data_holder=valid_data_holder)
        #
        # self.col_prefix_proxy \
        #     = ColPrefixNetProxy(base_net, predict_mode=predict_mode,
        #                         train_data_holder=train_data_holder, valid_data_holder=valid_data_holder)
        #
        # self.col_agg_proxy \
        #     = ColAggNetProxy(base_net, predict_mode=predict_mode,
        #                      train_data_holder=train_data_holder, valid_data_holder=valid_data_holder)
        #
        # self.cond_eq_proxy \
        #     = CondEqNetProxy(base_net, predict_mode=predict_mode,
        #                      train_data_holder=train_data_holder, valid_data_holder=valid_data_holder)
        #
        # self.cond_suffix_proxy \
        #     = CondSuffixNetProxy(base_net, predict_mode=predict_mode,
        #                          train_data_holder=train_data_holder, valid_data_holder=valid_data_holder)

    def run_a_epoch(self):
        self.n_col_proxy.run_a_epoch()
        # self.cond_prefix_proxy.run_a_epoch()
        # self.cond_operation_proxy.run_a_epoch()
        # self.cond_com_proxy.run_a_epoch()
        # self.cond_eq_proxy.run_a_epoch()
        # self.cond_suffix_proxy.run_a_epoch()
