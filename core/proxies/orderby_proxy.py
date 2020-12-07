from core.proxies.orderby_part.order_proxy import OrderNetProxy
from core.proxies.orderby_part.col_proxy import ColNetProxy


class OrderByProxy:
    def __init__(self, base_net, predict_mode=False, train_data_holder=None, valid_data_holder=None):
        self.mode = predict_mode
        self.order_proxy = OrderNetProxy(base_net, predict_mode=predict_mode,
                                         train_data_holder=train_data_holder, valid_data_holder=valid_data_holder)

        self.col_proxy \
            = ColNetProxy(base_net, predict_mode=predict_mode,
                          train_data_holder=train_data_holder, valid_data_holder=valid_data_holder)

    def run_a_epoch(self):
        self.order_proxy.run_a_epoch()
        self.col_proxy.run_a_epoch()
