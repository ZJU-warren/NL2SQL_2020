import numpy as np


class DataHolder:
    def __init__(self,  X):
        self.input_ids = np.array(X['input_ids'])
        self.token_type_ids = np.array(X['token_type_ids'])
        self.attention_mask = np.array(X['attention_mask'])
        self.tables = np.array(X['idx']['tables'], dtype='object')
        self.columns = np.array(X['idx']['columns'], dtype='object')

    def col_num(self, i):
        return sum([len(col) for col in self.columns[i]])