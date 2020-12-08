from core.proxies.limit_part.need_proxy import NeedNetProxy


class LimitProxy:
    def __init__(self, base_net, predict_mode=False, train_data_holder=None, valid_data_holder=None, test_data_holder=None):
        self.mode = predict_mode
        self.base_net = base_net
        self.train_data_holder = train_data_holder
        self.valid_data_holder = valid_data_holder
        self.test_data_holder = test_data_holder

        if self.mode is False:
            self.need_proxy = NeedNetProxy(self.base_net, predict_mode=self.mode,
                                            train_data_holder=self.train_data_holder,
                                            valid_data_holder=self.valid_data_holder,
                                            test_data_holder=self.test_data_holder)

    def run_a_epoch(self):
        self.need_proxy.run_a_epoch()

    def predict(self):
        self.need_proxy = NeedNetProxy(self.base_net, predict_mode=self.mode,
                                       train_data_holder=self.train_data_holder,
                                       valid_data_holder=self.valid_data_holder,
                                       test_data_holder=self.test_data_holder)
        self.need_proxy.predict()
