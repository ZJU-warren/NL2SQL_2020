from core.proxies.from_part.module_proxies import NCondNetProxy, CondPrefixNetProxy


class FromProxy:
    def __init__(self, base_net, predict_mode=False):
        self.mode = predict_mode
        # self.n_cond_proxy = NCondNetProxy(base_net, predict_mode=False)
        self.cond_prefix_proxy = CondPrefixNetProxy(base_net, predict_mode=False)

    def run_a_epoch(self):
        # self.n_cond_proxy.run_a_epoch()
        self.cond_prefix_proxy.run_a_epoch()
