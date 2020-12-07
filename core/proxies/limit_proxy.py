from core.proxies.limit_part.need_proxy import NeedNetProxy


class LimitProxy:
    def __init__(self, base_net, predict_mode=False, train_data_holder=None, valid_data_holder=None):
        self.mode = predict_mode
        self.need_proxy = NeedNetProxy(base_net, predict_mode=predict_mode,
                                        train_data_holder=train_data_holder, valid_data_holder=valid_data_holder)

    def run_a_epoch(self):
        self.need_proxy.run_a_epoch()
