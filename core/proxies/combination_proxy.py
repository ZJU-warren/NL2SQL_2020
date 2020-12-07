from core.proxies.combination_part.combination_proxy import CombinationNetProxy


class CombinationProxy:
    def __init__(self, base_net, predict_mode=False, train_data_holder=None, valid_data_holder=None):
        self.mode = predict_mode
        self.combination_proxy = CombinationNetProxy(base_net, predict_mode=predict_mode,
                                         train_data_holder=train_data_holder, valid_data_holder=valid_data_holder)

    def run_a_epoch(self):
        self.combination_proxy.run_a_epoch()
