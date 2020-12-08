from core.proxies.where_part.n_cond_proxy import WhereNCondNetProxy
from core.proxies.where_part.cond_operation_proxy import WhereCondOperationNetProxy
from core.proxies.where_part.cond_com_proxy import WhereCondComNetProxy
from core.proxies.where_part.cond_eq_proxy import WhereCondEqNetProxy
from core.proxies.where_part.cond_prefix_proxy import WhereCondPrefixNetProxy
from core.proxies.where_part.cond_suffix_proxy import WhereCondSuffixNetProxy


class WhereProxy:
    def __init__(self, base_net, predict_mode=False, train_data_holder=None, valid_data_holder=None, test_data_holder=None):
        self.mode = predict_mode
        self.base_net = base_net
        self.train_data_holder = train_data_holder
        self.valid_data_holder = valid_data_holder
        self.test_data_holder = test_data_holder

        if self.mode is False:
            self.n_cond_proxy = WhereNCondNetProxy(self.base_net, predict_mode=self.mode,
                                                   train_data_holder=self.train_data_holder,
                                                   valid_data_holder=self.valid_data_holder,
                                                   test_data_holder=self.test_data_holder)

            self.cond_prefix_proxy \
                = WhereCondPrefixNetProxy(self.base_net, predict_mode=self.mode,
                                          train_data_holder=self.train_data_holder,
                                          valid_data_holder=self.valid_data_holder,
                                          test_data_holder=self.test_data_holder)

            self.cond_operation_proxy \
                = WhereCondOperationNetProxy(self.base_net, predict_mode=self.mode,
                                             train_data_holder=self.train_data_holder,
                                             valid_data_holder=self.valid_data_holder,
                                             test_data_holder=self.test_data_holder)

            self.cond_com_proxy \
                = WhereCondComNetProxy(self.base_net, predict_mode=self.mode,
                                       train_data_holder=self.train_data_holder,
                                       valid_data_holder=self.valid_data_holder,
                                       test_data_holder=self.test_data_holder)

            self.cond_eq_proxy \
                = WhereCondEqNetProxy(self.base_net, predict_mode=self.mode,
                                      train_data_holder=self.train_data_holder,
                                      valid_data_holder=self.valid_data_holder,
                                      test_data_holder=self.test_data_holder)

            self.cond_suffix_proxy \
                = WhereCondSuffixNetProxy(self.base_net, predict_mode=self.mode,
                                          train_data_holder=self.train_data_holder,
                                          valid_data_holder=self.valid_data_holder,
                                          test_data_holder=self.test_data_holder)

    def run_a_epoch(self):
        # self.n_cond_proxy.run_a_epoch()
        # self.cond_prefix_proxy.run_a_epoch()
        # self.cond_operation_proxy.run_a_epoch()
        print("{}{}")
        self.cond_com_proxy.run_a_epoch()
        # self.cond_eq_proxy.run_a_epoch()
        # self.cond_suffix_proxy.run_a_epoch()

    def predict(self):
        self.n_cond_proxy = WhereNCondNetProxy(self.base_net, predict_mode=self.mode,
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

