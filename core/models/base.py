from core.models.others.bert import BertEncoder
import torch
from GlobalParameters import cuda_id, inf, max_columns_number, max_table_column_len


class Base(torch.nn.Module):
    def __init__(self, hidden=768, gpu=True):
        super(Base, self).__init__()

        self.bert = BertEncoder()
        self.gpu = gpu

        self.hidden = hidden
        self.att_matrix = torch.nn.Linear(self.hidden, self.hidden)
        self.u = torch.nn.Linear(self.hidden, self.hidden)
        self.v = torch.nn.Linear(self.hidden, self.hidden)
        self.soft_max = torch.nn.Softmax(dim=-1)

        if gpu:
            self.cuda(cuda_id)

    def forward(self, data_holder, X_id):
        batch_input_ids = torch.from_numpy(data_holder.input_ids[X_id])
        batch_token_type_ids = data_holder.token_type_ids[X_id]
        batch_attention_mask = data_holder.attention_mask[X_id]

        (tokens_embeddings, nsp_csl) \
            = self.bert(batch_input_ids, batch_token_type_ids, batch_attention_mask)
        cls = tokens_embeddings[:, 0]

        out, col_att = self.attention_unit(X_id.shape[0], data_holder, X_id, tokens_embeddings, cls)
        return cls, out, col_att

    def attention_unit(self, batch, data_holder, X_id, tokens_embeddings, cls):
        # generate r (columns_number * embedding_size) with attention
        columns_matrix = torch.zeros([batch, max_columns_number, max_table_column_len, self.hidden])
        table_list = data_holder.tables[X_id]
        column_list = data_holder.columns[X_id]
        batch_col_mask = []

        for i in range(batch):
            tables = table_list[i]
            columns = column_list[i]
            col = 0
            col_mask = []
            for j in range(len(tables)):
                tb = tables[j]
                l = tb[1] - tb[0]
                for k in range(len(columns[j])):
                    cl = columns[j][k]
                    col_mask.append(l + cl[1] - cl[0])
                    columns_matrix[i, col, 0:l, :] = tokens_embeddings[i, tb[0]:tb[1], :]
                    columns_matrix[i, col, l:l + cl[1] - cl[0], :] = tokens_embeddings[i, cl[0]:cl[1], :]
                    col += 1
            batch_col_mask.append(col_mask)

        if self.gpu:
            columns_matrix = columns_matrix.cuda(cuda_id)
        att_weight = (cls.unsqueeze(1).unsqueeze(1) * self.att_matrix(columns_matrix)).sum(-1)

        # use attention mask
        for i in range(batch):
            n_cols = len(batch_col_mask[i])
            for j in range(n_cols):
                att_weight[i, j, batch_col_mask[i][j]:] = -inf
            att_weight[i, n_cols:, :] = -inf

        # get attention score
        att_weight = self.soft_max(att_weight).unsqueeze(-1)
        att_columns = (att_weight * columns_matrix).sum(2)
        out = self.u(att_columns) + self.v(cls.unsqueeze(1))
        return out, att_columns

