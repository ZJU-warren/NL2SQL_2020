from core.proxies.having_part.n_cond_proxy import HavingNCondNetProxy
from core.proxies.having_part.cond_operation_proxy import HavingCondOperationNetProxy
from core.proxies.having_part.cond_com_proxy import HavingCondComNetProxy
from core.proxies.having_part.cond_eq_proxy import HavingCondEqNetProxy
from core.proxies.having_part.cond_prefix_proxy import HavingCondPrefixNetProxy
from core.proxies.having_part.cond_suffix_proxy import HavingCondSuffixNetProxy
from core.proxies.having_part.cond_agg_proxy import HavingCondAggNetProxy


class HavingProxy:
    def __init__(self, base_net, predict_mode=False, train_data_holder=None, valid_data_holder=None, test_data_holder=None):
        self.mode = predict_mode
        self.base_net = base_net
        self.train_data_holder = train_data_holder
        self.valid_data_holder = valid_data_holder
        self.test_data_holder = test_data_holder
        self.n_cond_proxy = HavingNCondNetProxy

        if self.mode is False:
            self.n_cond_proxy = HavingNCondNetProxy(self.base_net, predict_mode=self.mode,
                                                    train_data_holder=self.train_data_holder,
                                                    valid_data_holder=self.valid_data_holder,
                                                    test_data_holder=self.test_data_holder)

            self.cond_prefix_proxy \
                = HavingCondPrefixNetProxy(self.base_net, predict_mode=self.mode,
                                           train_data_holder=self.train_data_holder,
                                           valid_data_holder=self.valid_data_holder,
                                           test_data_holder=self.test_data_holder, thres=0.9)

            self.cond_agg_proxy \
                = HavingCondAggNetProxy(self.base_net, predict_mode=self.mode,
                                        train_data_holder=self.train_data_holder,
                                        valid_data_holder=self.valid_data_holder,
                                        test_data_holder=self.test_data_holder)

            self.cond_operation_proxy \
                = HavingCondOperationNetProxy(self.base_net, predict_mode=self.mode,
                                              train_data_holder=self.train_data_holder,
                                              valid_data_holder=self.valid_data_holder,
                                              test_data_holder=self.test_data_holder)

            self.cond_com_proxy \
                = HavingCondComNetProxy(self.base_net, predict_mode=self.mode,
                                        train_data_holder=self.train_data_holder,
                                        valid_data_holder=self.valid_data_holder,
                                        test_data_holder=self.test_data_holder)

            self.cond_eq_proxy \
                = HavingCondEqNetProxy(self.base_net, predict_mode=self.mode,
                                       train_data_holder=self.train_data_holder,
                                       valid_data_holder=self.valid_data_holder,
                                       test_data_holder=self.test_data_holder)

            self.cond_suffix_proxy \
                = HavingCondSuffixNetProxy(self.base_net, predict_mode=self.mode,
                                           train_data_holder=self.train_data_holder,
                                           valid_data_holder=self.valid_data_holder,
                                           test_data_holder=self.test_data_holder)

    def run_a_epoch(self):
        # self.n_cond_proxy.run_a_epoch()
        self.cond_prefix_proxy.run_a_epoch()
        # self.cond_agg_proxy.run_a_epoch()
        # self.cond_operation_proxy.run_a_epoch()
        # self.cond_com_proxy.run_a_epoch()
        # self.cond_eq_proxy.run_a_epoch()
        # self.cond_suffix_proxy.run_a_epoch()

    def predict(self):
        self.n_cond_proxy = HavingNCondNetProxy(self.base_net, predict_mode=self.mode,
                                                train_data_holder=self.train_data_holder,
                                                valid_data_holder=self.valid_data_holder,
                                                test_data_holder=self.test_data_holder)
        self.n_cond_proxy.predict()

        # self.cond_prefix_proxy \
        #     = HavingCondPrefixNetProxy(self.base_net, predict_mode=self.mode,
        #                          train_data_holder=self.train_data_holder,
        #                          valid_data_holder=self.valid_data_holder,
        #                          test_data_holder=self.test_data_holder)
        # self.cond_prefix_proxy.predict()
        #
        # self.cond_agg_proxy \
        #     = HavingCondAggNetProxy(self.base_net, predict_mode=self.mode,
        #                       train_data_holder=self.train_data_holder,
        #                       valid_data_holder=self.valid_data_holder,
        #                       test_data_holder=self.test_data_holder)
        # self.cond_agg_proxy.predict()
        #
        # self.cond_operation_proxy \
        #     = HavingCondOperationNetProxy(self.base_net, predict_mode=self.mode,
        #                             train_data_holder=self.train_data_holder,
        #                             valid_data_holder=self.valid_data_holder,
        #                             test_data_holder=self.test_data_holder)
        # self.cond_operation_proxy.predict()
        #
        # self.cond_com_proxy \
        #     = HavingCondComNetProxy(self.base_net, predict_mode=self.mode,
        #                       train_data_holder=self.train_data_holder,
        #                       valid_data_holder=self.valid_data_holder,
        #                       test_data_holder=self.test_data_holder)
        # self.cond_com_proxy.predict()
        #
        # self.cond_eq_proxy \
        #     = HavingCondEqNetProxy(self.base_net, predict_mode=self.mode,
        #                      train_data_holder=self.train_data_holder,
        #                      valid_data_holder=self.valid_data_holder,
        #                      test_data_holder=self.test_data_holder)
        # self.cond_eq_proxy.predict()
        #
        # self.cond_suffix_proxy \
        #     = HavingCondSuffixNetProxy(self.base_net, predict_mode=self.mode,
        #                          train_data_holder=self.train_data_holder,
        #                          valid_data_holder=self.valid_data_holder,
        #                          test_data_holder=self.test_data_holder)
        # self.cond_suffix_proxy.predict()
