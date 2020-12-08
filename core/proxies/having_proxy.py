from core.proxies.having_part.n_cond_proxy import NCondNetProxy
from core.proxies.having_part.cond_operation_proxy import CondOperationNetProxy
from core.proxies.having_part.cond_com_proxy import CondComNetProxy
from core.proxies.having_part.cond_eq_proxy import CondEqNetProxy
from core.proxies.having_part.cond_prefix_proxy import CondPrefixNetProxy
from core.proxies.having_part.cond_suffix_proxy import CondSuffixNetProxy
from core.proxies.having_part.cond_agg_proxy import CondAggNetProxy


class HavingProxy:
    def __init__(self, base_net, predict_mode=False, train_data_holder=None, valid_data_holder=None, test_data_holder=None):
        self.mode = predict_mode
        self.base_net = base_net
        self.train_data_holder = train_data_holder
        self.valid_data_holder = valid_data_holder
        self.test_data_holder = test_data_holder
        self.n_cond_proxy = NCondNetProxy

        if self.mode is False:
            self.n_cond_proxy = NCondNetProxy(self.base_net, predict_mode=self.mode,
                                            train_data_holder=self.train_data_holder,
                                            valid_data_holder=self.valid_data_holder,
                                            test_data_holder=self.test_data_holder)

            self.cond_prefix_proxy \
                = CondPrefixNetProxy(self.base_net, predict_mode=self.mode,
                                    train_data_holder=self.train_data_holder,
                                    valid_data_holder=self.valid_data_holder,
                                    test_data_holder=self.test_data_holder)

            self.cond_agg_proxy \
                = CondAggNetProxy(self.base_net, predict_mode=self.mode,
                                    train_data_holder=self.train_data_holder,
                                    valid_data_holder=self.valid_data_holder,
                                    test_data_holder=self.test_data_holder)

            self.cond_operation_proxy \
                = CondOperationNetProxy(self.base_net, predict_mode=self.mode,
                                 train_data_holder=self.train_data_holder,
                                 valid_data_holder=self.valid_data_holder,
                                 test_data_holder=self.test_data_holder)

            self.cond_com_proxy \
                = CondComNetProxy(self.base_net, predict_mode=self.mode,
                                 train_data_holder=self.train_data_holder,
                                 valid_data_holder=self.valid_data_holder,
                                 test_data_holder=self.test_data_holder)

            self.cond_eq_proxy \
                = CondEqNetProxy(self.base_net, predict_mode=self.mode,
                                  train_data_holder=self.train_data_holder,
                                  valid_data_holder=self.valid_data_holder,
                                  test_data_holder=self.test_data_holder)

            self.cond_suffix_proxy \
                = CondSuffixNetProxy(self.base_net, predict_mode=self.mode,
                                    train_data_holder=self.train_data_holder,
                                    valid_data_holder=self.valid_data_holder,
                                    test_data_holder=self.test_data_holder)

    def run_a_epoch(self):
        self.n_cond_proxy.run_a_epoch()
        self.cond_prefix_proxy.run_a_epoch()
        self.cond_agg_proxy.run_a_epoch()
        self.cond_operation_proxy.run_a_epoch()
        self.cond_com_proxy.run_a_epoch()
        self.cond_eq_proxy.run_a_epoch()
        self.cond_suffix_proxy.run_a_epoch()

    def predict(self):
        self.n_cond_proxy = NCondNetProxy(self.base_net, predict_mode=self.mode,
                                          train_data_holder=self.train_data_holder,
                                          valid_data_holder=self.valid_data_holder,
                                          test_data_holder=self.test_data_holder)
        self.n_cond_proxy.predict()

        # self.cond_prefix_proxy \
        #     = CondPrefixNetProxy(self.base_net, predict_mode=self.mode,
        #                          train_data_holder=self.train_data_holder,
        #                          valid_data_holder=self.valid_data_holder,
        #                          test_data_holder=self.test_data_holder)
        # self.cond_prefix_proxy.predict()
        #
        # self.cond_agg_proxy \
        #     = CondAggNetProxy(self.base_net, predict_mode=self.mode,
        #                       train_data_holder=self.train_data_holder,
        #                       valid_data_holder=self.valid_data_holder,
        #                       test_data_holder=self.test_data_holder)
        # self.cond_agg_proxy.predict()
        #
        # self.cond_operation_proxy \
        #     = CondOperationNetProxy(self.base_net, predict_mode=self.mode,
        #                             train_data_holder=self.train_data_holder,
        #                             valid_data_holder=self.valid_data_holder,
        #                             test_data_holder=self.test_data_holder)
        # self.cond_operation_proxy.predict()
        #
        # self.cond_com_proxy \
        #     = CondComNetProxy(self.base_net, predict_mode=self.mode,
        #                       train_data_holder=self.train_data_holder,
        #                       valid_data_holder=self.valid_data_holder,
        #                       test_data_holder=self.test_data_holder)
        # self.cond_com_proxy.predict()
        #
        # self.cond_eq_proxy \
        #     = CondEqNetProxy(self.base_net, predict_mode=self.mode,
        #                      train_data_holder=self.train_data_holder,
        #                      valid_data_holder=self.valid_data_holder,
        #                      test_data_holder=self.test_data_holder)
        # self.cond_eq_proxy.predict()
        #
        # self.cond_suffix_proxy \
        #     = CondSuffixNetProxy(self.base_net, predict_mode=self.mode,
        #                          train_data_holder=self.train_data_holder,
        #                          valid_data_holder=self.valid_data_holder,
        #                          test_data_holder=self.test_data_holder)
        # self.cond_suffix_proxy.predict()
