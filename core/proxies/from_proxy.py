from core.proxies.from_part.n_cond_proxy import FromNCondNetProxy
from core.proxies.from_part.j_cond_proxy import FromJCondNetProxy
from core.proxies.from_part.cond_prefix_proxy import FromCondPrefixNetProxy
from core.proxies.from_part.cond_suffix_proxy import FromCondSuffixNetProxy


class FromProxy:
    def __init__(self, base_net, predict_mode=False, train_data_holder=None, valid_data_holder=None, test_data_holder=None):
        self.mode = predict_mode
        self.base_net = base_net
        self.n_col_proxy = self.col_prefix_proxy = self.col_agg_proxy = self.col_com_proxy = self.col_suffix_proxy = None
        self.train_data_holder = train_data_holder
        self.valid_data_holder = valid_data_holder
        self.test_data_holder = test_data_holder

        if self.mode is False:
            self.n_cond_proxy = FromNCondNetProxy(self.base_net, predict_mode=self.mode,
                                                  train_data_holder=self.train_data_holder,
                                                  valid_data_holder=self.valid_data_holder,
                                                  test_data_holder=self.test_data_holder)

            self.cond_prefix_proxy \
                = FromCondPrefixNetProxy(self.base_net, predict_mode=self.mode,
                                         train_data_holder=self.train_data_holder,
                                         valid_data_holder=self.valid_data_holder,
                                         test_data_holder=self.test_data_holder)

            self.cond_suffix_proxy \
                = FromCondSuffixNetProxy(self.base_net, predict_mode=self.mode,
                                         train_data_holder=self.train_data_holder,
                                         valid_data_holder=self.valid_data_holder,
                                         test_data_holder=self.test_data_holder)

            self.j_cond_proxy = FromJCondNetProxy(self.base_net, predict_mode=self.mode,
                                                  train_data_holder=self.train_data_holder,
                                                  valid_data_holder=self.valid_data_holder,
                                                  test_data_holder=self.test_data_holder)

    def run_a_epoch(self):
        self.n_cond_proxy.run_a_epoch()
        # self.cond_prefix_proxy.run_a_epoch()
        # self.cond_suffix_proxy.run_a_epoch()
        # self.j_cond_proxy.run_a_epoch()

    def predict(self):
        self.n_cond_proxy = FromNCondNetProxy(self.base_net, predict_mode=self.mode,
                                              train_data_holder=self.train_data_holder,
                                              valid_data_holder=self.valid_data_holder,
                                              test_data_holder=self.test_data_holder)
        self.n_cond_proxy.predict()

        # self.cond_prefix_proxy \
        #     = CondPrefixNetProxy(self.base_net, predict_mode=self.mode,
        #                         train_data_holder=self.train_data_holder,
        #                         valid_data_holder=self.valid_data_holder,
        #                         test_data_holder=self.test_data_holder)
        # self.cond_prefix_proxy.predict()
        #
        # self.cond_suffix_proxy \
        #     = CondSuffixNetProxy(self.base_net, predict_mode=self.mode,
        #                      train_data_holder=self.train_data_holder,
        #                      valid_data_holder=self.valid_data_holder,
        #                      test_data_holder=self.test_data_holder)
        # self.cond_suffix_proxy.predict()
        #
        # self.j_cond_proxy = JCondNetProxy(self.base_net, predict_mode=self.mode,
        #                      train_data_holder=self.train_data_holder,
        #                      valid_data_holder=self.valid_data_holder,
        #                      test_data_holder=self.test_data_holder)
        # self.j_cond_proxy.predict()
