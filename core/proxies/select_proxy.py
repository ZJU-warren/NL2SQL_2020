from core.proxies.select_part.n_col_proxy import NColNetProxy
from core.proxies.select_part.col_prefix_proxy import ColPrefixNetProxy
from core.proxies.select_part.col_agg_proxy import ColAggNetProxy
from core.proxies.select_part.col_com_proxy import ColComNetProxy
from core.proxies.select_part.col_suffix_proxy import ColSuffixNetProxy


class SelectProxy:
    def __init__(self, base_net, predict_mode=False, train_data_holder=None, valid_data_holder=None, test_data_holder=None):
        self.mode = predict_mode
        self.n_col_proxy = NColNetProxy(base_net, predict_mode=predict_mode,
                                        train_data_holder=train_data_holder,
                                        valid_data_holder=valid_data_holder,
                                        test_data_holder=test_data_holder)

        self.col_prefix_proxy \
            = ColPrefixNetProxy(base_net, predict_mode=predict_mode,
                                train_data_holder=train_data_holder,
                                valid_data_holder=valid_data_holder,
                                test_data_holder=test_data_holder)

        self.col_agg_proxy \
            = ColAggNetProxy(base_net, predict_mode=predict_mode,
                             train_data_holder=train_data_holder,
                             valid_data_holder=valid_data_holder,
                             test_data_holder=test_data_holder)

        self.col_com_proxy \
            = ColComNetProxy(base_net, predict_mode=predict_mode,
                             train_data_holder=train_data_holder,
                             valid_data_holder=valid_data_holder,
                             test_data_holder=test_data_holder)

        self.col_suffix_proxy \
            = ColSuffixNetProxy(base_net, predict_mode=predict_mode,
                                train_data_holder=train_data_holder,
                                valid_data_holder=valid_data_holder,
                                test_data_holder=test_data_holder)

    def run_a_epoch(self):
        self.n_col_proxy.run_a_epoch()
        self.col_prefix_proxy.run_a_epoch()
        self.col_agg_proxy.run_a_epoch()
        self.col_com_proxy.run_a_epoch()
        self.col_suffix_proxy.run_a_epoch()

    def predict(self):
        self.n_col_proxy.predict()
        self.col_prefix_proxy.predict()
        self.col_agg_proxy.predict()
        self.col_com_proxy.predict()
        self.col_suffix_proxy.predict()
