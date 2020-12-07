import DataLinkSet as DLSet
import json
import numpy as np
from GlobalParameters import learning_rate
import torch.optim as optim
from tools.metrics import acc
import random
import torch


class ModuleProxy:
    def _init_env(self, base_net, target_net, part_name, file_name, tensor=False):
        if self.mode:
            self._init_test(base_net, target_net, part_name, file_name, tensor=tensor)

        else:
            self._init_train(base_net, target_net, part_name, file_name, tensor=tensor)

    def _init_test(self, base_net, target_net, part_name, file_name, tensor=False):
        # init model
        self.target_net = target_net(base_net)
        self.load_model()

        # init data
        with open(DLSet.main_folder_link % 'Test' + '/%s/X_pd_sup_%s' % (part_name, file_name), 'r') as f:
            info = json.load(f)
            self.X_id = np.array(info['X_id'], dtype=np.int32)

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
            self.valid_X_id = np.array(json.load(f)['X_id'], dtype=np.int32)

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
        self.best_acc = 0
        self.last_acc = 0
        self.avg_loss = 0
        self.optimizer = None
        self.loss = 0
        self.step = 0
        self.start = 0

    def predict(self, top=1):
        y_pd_score = self.target_net(self.train_data_holder, self.X_id)
        print(y_pd_score)

    def run_a_epoch(self):
        # only for train

        self.last_acc = 0
        self.avg_loss = 0
        step = 0

        while True:
            if self.step == 0:
                self.last_acc *= step

            # calculate the start, end of this batch
            end = min(self.total, self.start + self.batch_size)
            # print('[%d, %d)' % (self.start, end))
            data_index = list(range(self.start, end))

            # forward
            self.forward(data_index)

            # step
            if self.step == 0:
                step += 1
                self.last_acc /= step

            # update for next batch
            self.start = end % self.total

            if self.start == 0:
                break

        if self.last_acc > 0.8 and self.best_acc < self.last_acc:
            print('=============== save the best model [%s] with acc %f ================='
                  % (self.__class__.__name__, self.last_acc))
            self.save_model()
            self.best_acc = self.last_acc
            # self.load_model()

        print('- [%s] with loss %f and acc %f in the last epoch.'
              % (self.__class__.__name__, self.avg_loss, self.last_acc))

    def forward(self, data_index):
        y_pd_score = self.target_net(self.train_data_holder, self.X_id[data_index])

        if self.mode is False:
            self.backward(y_pd_score, data_index, None)

    def backward(self, y_pd, data_index, loss, top=None):
        self.avg_loss = (self.avg_loss * self.step + loss.data.cpu().numpy()) / (self.step + 1)

        self.step += 1
        self.loss = loss
        self.loss.backward()

        if self.step % 20 == 0:
            print('-- loss_cpu', self.loss.data.cpu().numpy())
            self.optimizer.step()
            self.optimizer.zero_grad()

            # calculate acc
            # @train
            gt = self.y_gt[data_index]
            if top is None:
                acc_value = acc(y_pd.data.cpu().numpy(), gt)
            else:
                acc_value = acc(y_pd.data.cpu().numpy(), gt, top=top[0][data_index])
            print('%s -- acc@train' % self.__class__.__name__, acc_value)

            # @validation
            total_valid = len(self.valid_y_gt)
            data_index = random.sample([i for i in range(total_valid)], 10)
            gt = self.valid_y_gt[data_index]
            y_pd_valid = self.target_net(self.valid_data_holder, self.valid_X_id[data_index])

            if top is None:
                acc_value_valid = acc(y_pd_valid.data.cpu().numpy(), gt)
            else:
                acc_value_valid = acc(y_pd_valid.data.cpu().numpy(), gt, top=top[1][data_index])

            print('%s -- acc@valid' % self.__class__.__name__, acc_value_valid)
            self.last_acc += acc_value_valid
            self.step = 0

    def save_model(self):
        # stash
        base_net = self.target_net.base_net
        self.target_net.base_net = None
        torch.save(self.target_net.state_dict(), DLSet.model_folder_link + '/%s' % self.__class__.__name__)

        # recover
        self.target_net.base_net = base_net

    def load_model(self):
        # stash
        base_net = self.target_net.base_net
        self.target_net.load(DLSet.model_folder_link + '/%s' % self.__class__.__name__)

        # recover
        self.target_net.base_net = base_net
