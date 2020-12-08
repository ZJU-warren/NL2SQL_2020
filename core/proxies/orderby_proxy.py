from core.proxies.orderby_part.order_proxy import OrderByOrderNetProxy
from core.proxies.orderby_part.col_proxy import OrderByColNetProxy


class OrderByProxy:
    def __init__(self, base_net, predict_mode=False, train_data_holder=None, valid_data_holder=None, test_data_holder=None):
        self.mode = predict_mode
        self.base_net = base_net
        self.train_data_holder = train_data_holder
        self.valid_data_holder = valid_data_holder
        self.test_data_holder = test_data_holder
        if self.mode is False:
            self.order_proxy = OrderByOrderNetProxy(self.base_net, predict_mode=self.mode,
                                                    train_data_holder=self.train_data_holder,
                                                    valid_data_holder=self.valid_data_holder,
                                                    test_data_holder=self.test_data_holder)

            self.col_proxy \
                = OrderByColNetProxy(self.base_net, predict_mode=self.mode,
                                     train_data_holder=self.train_data_holder,
                                     valid_data_holder=self.valid_data_holder,
                                     test_data_holder=self.test_data_holder)

    def run_a_epoch(self):
        self.order_proxy.run_a_epoch()
        self.col_proxy.run_a_epoch()

    def predict(self):
        self.order_proxy = OrderByOrderNetProxy(self.base_net, predict_mode=self.mode,
                                                train_data_holder=self.train_data_holder,
                                                valid_data_holder=self.valid_data_holder,
                                                test_data_holder=self.test_data_holder)
        self.order_proxy.predict()

        self.col_proxy \
            = OrderByColNetProxy(self.base_net, predict_mode=self.mode,
                                 train_data_holder=self.train_data_holder,
                                 valid_data_holder=self.valid_data_holder,
                                 test_data_holder=self.test_data_holder)
        self.col_proxy.predict()