import DataLinkSet as DLSet
import json
import numpy as np
from GlobalParameters import learning_rate
import torch.optim as optim
from tools.metrics import acc
import random
import torch


class ModuleProxy:
    def _init_train(self, base_net, target_net, part_name, file_name, tensor=False):
        # init model
        self.target_net = target_net(base_net)

        # init data
        with open(DLSet.main_folder_link % 'Train' + '/%s/X_gt_sup_%s' % (part_name, file_name), 'r') as f:
            info = json.load(f)
            self.X_id = np.array(info['X_id'], dtype=np.int32)

        with open(DLSet.main_folder_link % 'Train' + '/%s/y_gt_%s' % (part_name, file_name), 'r') as f:
            if tensor:
                self.y_gt = torch.Tensor(json.load(f)[file_name]).long()
            else:
                self.y_gt = np.array(json.load(f)[file_name], dtype=object)

        with open(DLSet.main_folder_link % 'Validation' + '/%s/X_gt_sup_%s' % (part_name, file_name), 'r') as f:
            info = json.load(f)
            self.valid_X_id = np.array(info['X_id'], dtype=np.int32)

        with open(DLSet.main_folder_link % 'Validation' + '/%s/y_gt_%s' % (part_name, file_name), 'r') as f:
            if tensor:
                self.valid_y_gt = torch.Tensor(json.load(f)[file_name]).long()
            else:
                self.valid_y_gt = np.array(json.load(f)[file_name], dtype=object)

        # epoch init
        self.total = self.y_gt.shape[0]
        self.start = 0
        self.batch_size = max(self.total // self.batch, 5)
        self.optimizer = optim.Adam(self.target_net.parameters(), lr=learning_rate)

    def __init__(self, predict_mode=False, train_data_holder=None, valid_data_holder=None, batch=1000):
        self.mode = predict_mode
        self.batch = batch
        self.train_data_holder = train_data_holder
        self.valid_data_holder = valid_data_holder

        self.optimizer = None
        self.loss = 0
        self.step = 0
        self.start = 0

    def run_a_epoch(self):
        if self.mode:
            data_index = list(range(0, self.total))
            self.forward(data_index)

        else:
            while True:
                # calculate the start, end of this batch
                end = min(self.total, self.start + self.batch_size)
                # print('[%d, %d)' % (self.start, end))
                data_index = list(range(self.start, end))

                # forward
                self.forward(data_index)

                # update for next batch
                self.start = end % self.total

                if self.start == 0:
                    break

    def forward(self, data_index):
        y_pd_suffix_score = self.target_net(self.train_data_holder, self.X_id[data_index])

        if self.mode is False:
            self.backward(y_pd_suffix_score, data_index, None)

    def backward(self, y_pd, data_index, loss, top=None):
        self.step += 1
        self.loss = loss
        self.loss.backward()

        if self.step % 10 == 0:
            print('-- loss_cpu', self.loss.data.cpu().numpy())
            self.optimizer.step()
            self.optimizer.zero_grad()

            # calculate acc
            # @train
            gt = self.y_gt[data_index]
            if top is None:
                print('%s -- acc@train' % self.__class__.__name__, acc(y_pd.data.cpu().numpy(), gt))
            else:
                print('%s -- acc@train' % self.__class__.__name__,
                      acc(y_pd.data.cpu().numpy(), gt, top=top[0][data_index]))

            # @validation
            total_valid = len(self.valid_y_gt)
            data_index = random.sample([i for i in range(total_valid)], 10)
            gt = self.valid_y_gt[data_index]
            y_pd_valid = self.target_net(self.valid_data_holder, self.valid_X_id[data_index])
            if top is None:
                print('%s -- acc@valid' % self.__class__.__name__, acc(y_pd_valid.data.cpu().numpy(), gt))
            else:
                print('%s -- acc@valid' % self.__class__.__name__,
                      acc(y_pd_valid.data.cpu().numpy(), gt, top=top[1][data_index]))
            self.step = 0



