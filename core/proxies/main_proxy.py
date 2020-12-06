from core.proxies.from_part.from_proxy import FromProxy
from core.models.base import Base
import json
import DataLinkSet as DLSet


class MainProxy:
    def __init_train(self):
        # load data
        with open(DLSet.X_link % 'Train', 'r') as f:
            self.X = json.load(f)

        # init model
        self.base_net = Base(self.X)

    def __init__(self, predict_mode=False, epoch=10):
        self.mode = predict_mode
        self.epoch = epoch

        if self.mode:
            pass
        else:
            self.__init_train()
            self.from_proxy = FromProxy(self.base_net, self.mode)

    def run(self):
        if self.mode:
            pass
        else:
            for _ in range(self.epoch):
                self.from_proxy.run_a_epoch()



