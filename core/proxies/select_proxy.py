from core.proxies.select_part.n_col_proxy import SelectNColNetProxy
from core.proxies.select_part.col_prefix_proxy import SelectColPrefixNetProxy
from core.proxies.select_part.col_agg_proxy import SelectColAggNetProxy
from core.proxies.select_part.col_com_proxy import SelectColComNetProxy
from core.proxies.select_part.col_suffix_proxy import SelectColSuffixNetProxy


class SelectProxy:
    def __init__(self, base_net, predict_mode=False, train_data_holder=None, valid_data_holder=None, test_data_holder=None):
        self.mode = predict_mode
        self.base_net = base_net
        self.n_col_proxy = self.col_prefix_proxy = self.col_agg_proxy = self.col_com_proxy = self.col_suffix_proxy = None
        self.train_data_holder = train_data_holder
        self.valid_data_holder = valid_data_holder
        self.test_data_holder = test_data_holder

        if self.mode is False:
            self.n_col_proxy = SelectNColNetProxy(self.base_net, predict_mode=self.mode,
                                                  train_data_holder=self.train_data_holder,
                                                  valid_data_holder=self.valid_data_holder,
                                                  test_data_holder=self.test_data_holder)

            self.col_prefix_proxy \
                = SelectColPrefixNetProxy(self.base_net, predict_mode=self.mode,
                                          train_data_holder=self.train_data_holder,
                                          valid_data_holder=self.valid_data_holder,
                                          test_data_holder=self.test_data_holder)

            self.col_agg_proxy \
                = SelectColAggNetProxy(self.base_net, predict_mode=self.mode,
                                       train_data_holder=self.train_data_holder,
                                       valid_data_holder=self.valid_data_holder,
                                       test_data_holder=self.test_data_holder)

            self.col_com_proxy \
                = SelectColComNetProxy(self.base_net, predict_mode=self.mode,
                                       train_data_holder=self.train_data_holder,
                                       valid_data_holder=self.valid_data_holder,
                                       test_data_holder=self.test_data_holder)

            self.col_suffix_proxy \
                = SelectColSuffixNetProxy(self.base_net, predict_mode=self.mode,
                                          train_data_holder=self.train_data_holder,
                                          valid_data_holder=self.valid_data_holder,
                                          test_data_holder=self.test_data_holder)

    def run_a_epoch(self):
        self.n_col_proxy.run_a_epoch()
        self.col_prefix_proxy.run_a_epoch()
        self.col_agg_proxy.run_a_epoch()
        self.col_com_proxy.run_a_epoch()
        self.col_suffix_proxy.run_a_epoch()

    def predict(self):
        # self.n_col_proxy = SelectNColNetProxy(self.base_net, predict_mode=self.mode,
        #                                       train_data_holder=self.train_data_holder,
        #                                       valid_data_holder=self.valid_data_holder,
        #                                       test_data_holder=self.test_data_holder)
        # self.n_col_proxy.predict()

        # self.col_prefix_proxy \
        #     = ColPrefixNetProxy(self.base_net, predict_mode=self.mode,
        #                         train_data_holder=self.train_data_holder,
        #                         valid_data_holder=self.valid_data_holder,
        #                         test_data_holder=self.test_data_holder)
        # self.col_prefix_proxy.predict()
        #
        self.col_agg_proxy \
            = SelectColAggNetProxy(self.base_net, predict_mode=self.mode,
                             train_data_holder=self.train_data_holder,
                             valid_data_holder=self.valid_data_holder,
                             test_data_holder=self.test_data_holder)
        self.col_agg_proxy.predict()
        #
        # self.col_com_proxy \
        #     = ColComNetProxy(self.base_net, predict_mode=self.mode,
        #                      train_data_holder=self.train_data_holder,
        #                      valid_data_holder=self.valid_data_holder,
        #                      test_data_holder=self.test_data_holder)
        # self.col_com_proxy.predict()
        #
        # self.col_suffix_proxy \
        #     = ColSuffixNetProxy(self.base_net, predict_mode=self.mode,
        #                         train_data_holder=self.train_data_holder,
        #                         valid_data_holder=self.valid_data_holder,
        #                         test_data_holder=self.test_data_holder)
        # self.col_suffix_proxy.predict()
