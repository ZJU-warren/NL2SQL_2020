import DataLinkSet as DLSet
import json
import numpy as np
from GlobalParameters import learning_rate, cuda_id
import torch.optim as optim
from tools.metrics import acc
from tools.prediction_gen import predict_generate
import random
import torch
import time


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
        self.X_id = np.array(range(self.test_data_holder.total))
        self.total = self.test_data_holder.total
        # print(self.__class__.__name__, self.total)

    def _init_train(self, base_net, target_net, part_name, file_name, tensor=False):
        # init model
        self.target_net = target_net(base_net)

        # init data
        with open(DLSet.main_folder_link % 'Train' + '/%s/X_gt_sup_%s' % (part_name, file_name), 'r') as f:
            self.X_id = np.array(json.load(f)['X_id'])

        with open(DLSet.main_folder_link % 'Train' + '/%s/y_gt_%s' % (part_name, file_name), 'r') as f:
            if tensor:
                self.y_gt = torch.Tensor(json.load(f)[file_name]).long()
            else:
                self.y_gt = np.array(json.load(f)[file_name], dtype=object)

        with open(DLSet.main_folder_link % 'Validation' + '/%s/X_gt_sup_%s' % (part_name, file_name), 'r') as f:
            self.valid_X_id = np.array(json.load(f)['X_id'])

        with open(DLSet.main_folder_link % 'Validation' + '/%s/y_gt_%s' % (part_name, file_name), 'r') as f:
            if tensor:
                self.valid_y_gt = torch.Tensor(json.load(f)[file_name]).long()
            else:
                self.valid_y_gt = np.array(json.load(f)[file_name], dtype=object)

        # epoch init
        self.total = self.y_gt.shape[0]
        self.start = 0
        self.batch_size = 8
        self.optimizer = optim.Adam(self.target_net.parameters(), lr=learning_rate)

    def __init__(self, predict_mode=False, train_data_holder=None, valid_data_holder=None,
        test_data_holder=None, batch=1500, thres=0.92):

        self.thres = thres
        self.mode = predict_mode
        self.batch = batch
        self.train_data_holder = train_data_holder
        self.valid_data_holder = valid_data_holder
        self.test_data_holder = test_data_holder
        self.best_acc = 0
        self.last_acc = 0
        self.avg_loss = 0
        self.optimizer = None
        self.loss = 0
        self.step = 0
        self.start = 0
        self.need_train = True
        self.header_mask = None

    def predict(self, top=1, keyword=None, target_path=None, extra=None):
        y_pd_score = []

        i = 0
        while True:
            start = i * 20
            end = min((i + 1) * 20, self.total)

            if extra is None:
                score = self.target_net(self.test_data_holder,
                                                  self.X_id[start: end]).data.cpu().numpy()
            else:
                score = self.target_net(self.test_data_holder,
                                                  self.X_id[start: end], extra[start: end]).data.cpu().numpy()

            if self.header_mask is not None:
                y_pd_score.extend(score * self.header_mask[start: end].data.cpu().numpy())
            else:
                y_pd_score.extend(score)

            if end == self.total:
                break

            i += 1
            torch.cuda.empty_cache()

        result = predict_generate(np.array(y_pd_score), top)
        # generate prediction
        prediction = {
            'X_id': [],
            'question_id': [],
            keyword: [],
        }

        for _ in range(self.total):
            question_id = self.test_data_holder.get_question_id(self.X_id[_])
            prediction['X_id'].append(int(self.X_id[_].item()))
            prediction['question_id'].append(question_id)

            if type(result[_]) != np.ndarray:
                prediction[keyword].append(int(result[_]))
            else:
                prediction[keyword].append(result[_].tolist())

        with open(DLSet.result_folder_link + target_path, 'w') as f:
            f.write(json.dumps(prediction, ensure_ascii=False, indent=4, separators=(',', ':')))

        return result

    def run_a_epoch(self):
        print('Run %s, total = %d, batchsize = %d' % (self.__class__.__name__, self.total, self.batch_size))
        if self.need_train is False:
            return

        # only for train
        self.last_acc = 0
        self.avg_loss = 0

        step = 0
        while True:
            # calculate the start, end of this batch
            end = min(self.total, self.start + self.batch_size)
            # print('[%d, %d)' % (self.start, end))
            data_index = list(range(self.start, end))

            # forward
            acc_value_valid = self.forward(data_index)

            # step
            if acc_value_valid != -1:
                self.last_acc *= step
                step += 1
                self.last_acc += acc_value_valid
                self.last_acc /= step

            # update for next batch
            self.start = end % self.total

            if self.start == 0:
                break
            
        # print("____________thre %f", self.thres)
        # # if self.last_acc > 0.8 and self.best_acc < self.last_acc:
        # if self.last_acc > 0.4 and self.best_acc < self.last_acc:
        print('=============== save the best model [%s] with acc %f ================='
              % (self.__class__.__name__, self.last_acc))
        self.save_model()
        self.best_acc = self.last_acc
        # self.load_model()

            #if self.last_acc > self.thres:
        self.need_train = False

        print('- [%s] with loss %f and acc %f in the last epoch.'
              % (self.__class__.__name__, self.avg_loss, self.last_acc))

    def forward(self, data_index):
        y_pd_score = self.target_net(self.train_data_holder, self.X_id[data_index])
        return self.backward(y_pd_score, data_index, None)

    def backward(self, y_pd, data_index, loss, top=None):
        self.avg_loss = (self.avg_loss * self.step + loss.data.cpu().numpy()) / (self.step + 1)
        self.step += 1
        self.loss = loss
        self.loss.backward()

        acc_value_valid = -1

        if self.step % 10 == 0:
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
            data_index = random.sample([i for i in range(total_valid)], 8)

            gt = self.valid_y_gt[data_index]
            y_pd_valid = self.target_net(self.valid_data_holder, self.valid_X_id[data_index])

            if self.header_mask is not None:
                y_pd_valid = self.header_mask[data_index] * y_pd_valid

            if top is None:
                acc_value_valid = acc(y_pd_valid.data.cpu().numpy(), gt)
            else:
                acc_value_valid = acc(y_pd_valid.data.cpu().numpy(), gt, top=top[1][data_index])

            print('%s -- acc@valid' % self.__class__.__name__, acc_value_valid)
            self.step = 0

        return acc_value_valid

    def save_model(self):
        # stash
        base_net = self.target_net.base_net
        self.target_net.base_net = None
        torch.save(self.target_net.state_dict(), DLSet.model_folder_link + '/%s' % self.__class__.__name__)

        # recover
        self.target_net.base_net = base_net

    def load_model(self):
        # stash
        pre_trained_dict = torch.load(DLSet.model_folder_link + '/%s' % self.__class__.__name__)
        model_dict = self.target_net.state_dict()

        pre_trained_dict = {k: v
                            for k, v in pre_trained_dict.items()
                            if k in model_dict}
        model_dict.update(pre_trained_dict)
        self.target_net.load_state_dict(model_dict)

