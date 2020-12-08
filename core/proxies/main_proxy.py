from core.proxies.from_proxy import FromProxy
from core.proxies.where_proxy import WhereProxy
from core.proxies.groupby_proxy import GroupByProxy
from core.proxies.orderby_proxy import OrderByProxy
from core.proxies.combination_proxy import CombinationProxy
from core.proxies.select_proxy import SelectProxy
from core.proxies.limit_proxy import LimitProxy
from core.proxies.having_proxy import HavingProxy
from core.models.base import Base
import json
import DataLinkSet as DLSet
from core.data_holder import DataHolder
import torch


class MainProxy:
    def __init_train(self):
        # load data
        with open(DLSet.X_link % 'Train', 'r') as f:
            self.train_data_holder = DataHolder(json.load(f))

        with open(DLSet.X_link % 'Validation', 'r') as f:
            self.valid_data_holder = DataHolder(json.load(f))

    def __init_test(self):
        # load data
        with open(DLSet.X_link % 'Test', 'r') as f:
            self.test_data_holder = DataHolder(json.load(f))

    def __init__(self, predict_mode=False, epoch=100):
        self.mode = predict_mode
        self.epoch = epoch

        # init model
        self.base_net = Base()
        self.train_data_holder = None
        self.valid_data_holder = None
        self.test_data_holder = None

        if self.mode:
            self.load_model()
            self.__init_test()
        else:
            self.__init_train()

        self.select_proxy = SelectProxy(self.base_net, self.mode, self.train_data_holder, self.valid_data_holder, self.test_data_holder)
        self.from_proxy = FromProxy(self.base_net, self.mode, self.train_data_holder, self.valid_data_holder, self.test_data_holder)
        self.where_proxy = WhereProxy(self.base_net, self.mode, self.train_data_holder, self.valid_data_holder, self.test_data_holder)
        self.having_proxy = HavingProxy(self.base_net, self.mode, self.train_data_holder, self.valid_data_holder, self.test_data_holder)
        self.groupby_proxy = GroupByProxy(self.base_net, self.mode, self.train_data_holder, self.valid_data_holder, self.test_data_holder)
        self.orderby_proxy = OrderByProxy(self.base_net, self.mode, self.train_data_holder, self.valid_data_holder, self.test_data_holder)
        self.limit_proxy = LimitProxy(self.base_net, self.mode, self.train_data_holder, self.valid_data_holder, self.test_data_holder)
        self.combination_proxy = CombinationProxy(self.base_net, self.mode, self.train_data_holder, self.valid_data_holder, self.test_data_holder)

    def run(self):
        if self.mode:
            # self.select_proxy.predict()
            self.from_proxy.predict()
            self.where_proxy.predict()
            self.having_proxy.predict()
            self.groupby_proxy.predict()
            self.orderby_proxy.predict()
            self.limit_proxy.predict()
            self.combination_proxy.predict()
        else:
            for _ in range(self.epoch):
                # self.select_proxy.run_a_epoch()
                self.from_proxy.run_a_epoch()
                self.where_proxy.run_a_epoch()
                self.having_proxy.run_a_epoch()
                self.groupby_proxy.run_a_epoch()
                self.orderby_proxy.run_a_epoch()
                self.limit_proxy.run_a_epoch()
                self.combination_proxy.run_a_epoch()
            self.save_model()

    def save_model(self):
        torch.save(self.base_net.state_dict(), DLSet.model_folder_link + '/%s' % self.__class__)

    def load_model(self):
        self.base_net.load_state_dict(torch.load(DLSet.model_folder_link + '/%s' % self.__class__))
