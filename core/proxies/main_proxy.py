from core.proxies.from_proxy import FromProxy
from core.proxies.where_proxy import WhereProxy
from core.proxies.groupby_proxy import GroupByProxy
from core.proxies.orderby_proxy import OrderByProxy
from core.proxies.combination_proxy import CombinationProxy
from core.proxies.limit_proxy import LimitProxy
from core.models.base import Base
import json
import DataLinkSet as DLSet
from core.data_holder import DataHolder


class MainProxy:
    def __init_train(self):
        # load data
        with open(DLSet.X_link % 'Train', 'r') as f:
            self.train_data_holder = DataHolder(json.load(f))

        with open(DLSet.X_link % 'Validation', 'r') as f:
            self.valid_data_holder = DataHolder(json.load(f))

        # init model
        self.base_net = Base()

    def __init__(self, predict_mode=False, epoch=100):
        self.mode = predict_mode
        self.epoch = epoch

        if self.mode:
            pass
        else:
            self.__init_train()
            self.from_proxy = FromProxy(self.base_net, self.mode, self.train_data_holder, self.valid_data_holder)
            self.where_proxy = WhereProxy(self.base_net, self.mode, self.train_data_holder, self.valid_data_holder)
            self.having_proxy = WhereProxy(self.base_net, self.mode, self.train_data_holder, self.valid_data_holder)
            self.groupby_proxy = GroupByProxy(self.base_net, self.mode, self.train_data_holder, self.valid_data_holder)
            self.orderby_proxy = OrderByProxy(self.base_net, self.mode, self.train_data_holder, self.valid_data_holder)
            self.limit_proxy = LimitProxy(self.base_net, self.mode, self.train_data_holder, self.valid_data_holder)
            self.combination_proxy = CombinationProxy(self.base_net, self.mode, self.train_data_holder, self.valid_data_holder)

    def run(self):
        if self.mode:
            pass
        else:
            for _ in range(self.epoch):
                self.from_proxy.run_a_epoch()
                self.where_proxy.run_a_epoch()
                self.having_proxy.run_a_epoch()
                self.groupby_proxy.run_a_epoch()
                self.orderby_proxy.run_a_epoch()
                self.limit_proxy.run_a_epoch()
                self.combination_proxy.run_a_epoch()



