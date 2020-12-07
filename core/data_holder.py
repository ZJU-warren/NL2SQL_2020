import numpy as np
import torch
from GlobalParameters import cuda_id


class DataHolder:
    def __init__(self,  X):
        self.input_ids = np.array(X['input_ids'])
        self.token_type_ids = torch.Tensor(X['token_type_ids']).long().cuda(cuda_id)
        self.attention_mask = torch.Tensor(X['attention_mask']).long().cuda(cuda_id)
        self.tables = np.array(X['idx']['tables'], dtype='object')
        self.columns = np.array(X['idx']['columns'], dtype='object')
        self.question_id = X['question_id']

        self.total = len(self.question_id)

    def col_num(self, i):
        return sum([len(col) for col in self.columns[i]])

    def get_question_id(self, i):
        return self.question_id[i]
