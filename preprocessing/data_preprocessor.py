"""
Preprocess the data source to (X, y) for 'Train' and 'Validation' or X for 'Test'
"""

import json
import pandas as pd
import os
import DataLinkSet as DLSet
from transformers import BertTokenizer


class DataPreprocessor:
    tokenizer = BertTokenizer.from_pretrained(DLSet.vocab_path)

    @staticmethod
    def preprocess(data_source: str, cheat_mode: bool = True):
        """preprocess the given data source.

        :param data_source: the data sources need preprocess.
        :param cheat_mode: whether the data source contains ground truth.
        :return: None
        """
        print('-' * 20 + ' preprocess %s ' % data_source + '-' * 20)

        # load raw data
        data = json.load(DLSet.raw_data_link % data_source)
        content = json.load(DLSet.raw_content_link % data_source)
        schema = json.load(DLSet.raw_schema_link % data_source)

        # generate X after tokenizer
        DataPreprocessor.__generate_X(data, content, schema, DLSet.X_link % data_source)

    @staticmethod
    def __generate_db_info(schema: dict):
        """ load schema to a dict whose key is db_name
        and the value corresponded is the tables-columns information

        :param schema: schema loaded from raw data
        :return: db_info: a dict contains tables-columns information
        """

        db_info = dict()
        max_n_col = 0

        # for each database
        for each in schema:
            total_col = len(each['col_name'])
            focus_table = -1

            str_tables = '[unused1]当前时间[SEP]距今最近近期[unused1]所有相关数据总共[SEP]分别有哪些'
            n_col = 2

            # for each table in the database
            for i in range(1, total_col):
                # visit a new table
                if focus_table != int(each['col_name'][i][0].split('_')[-1]):
                    focus_table = int(each['col_name'][i][0].split('_')[-1])
                    str_tables += '[unused1]' + each['table_name'][focus_table]

                # visit a new column
                str_tables += '[SEP]' + each['col_name'][i][1]
                n_col += 1

            # stash this database
            db_info[each['db_name']] = str_tables
            max_n_col = max(max_n_col, n_col)

        # the max number of columns
        print('the max number of columns:', max_n_col)

        return db_info

    @staticmethod
    def __generate_X(data: dict, content: dict, schema: dict, store_link: str):
        """generate the X contains query, tables, columns information after tokenize.

        :param data: a dict contains db_name, question_id and question.
        :param content: content of database given by db_name in data.
        :param schema: schema of database given by db_name in data.
        :param store_link: the location to store the result.
        :return: None
        """
        # 1. get the toString information of each database
        db_info = DataPreprocessor.__generate_db_info(schema)

        # 2. combine query append with the corresponded database's information, and load others data
        query_with_db_info = []
        question_id = []
        db_name = []
        for each in data:
            query_with_db_info.append(each['question'] + db_info[each['db_name']])
            question_id.append(each['question_id'])
            db_name.append(each['db_name'])

        # 3. generate the tokens
        tokens = DataPreprocessor.tokenizer(query_with_db_info,
                                            padding=True, truncation=True, max_length=512, return_tensors="pt")

        # 4. generate X
        # generate target input_ids, token_type_ids, attention_mask
        input_ids = tokens['input_ids'].numpy().tolist()
        token_type_ids = tokens['token_type_ids'].numpy().tolist()
        attention_mask = tokens['attention_mask'].numpy().tolist()

        idx = {
            "tables": [],
            "columns": []
        }

        # generate the idx for each x
        max_len = -1
        max_id = -1
        cnt = 0

        # for each sample
        for x in input_ids:
            # calculate the tables and columns index for each x
            length = len(x)

            # calculate the start of table-columns
            table_start = x.index(140)

            # initialize the variables
            tables_index = []
            columns_index = []
            columns_index_one_table = []
            columns_start = -1

            for i in range(table_start, length + 1):
                # stash table
                if i + 1 == length or x[i + 1] == 0 or x[i] == 138:
                    # stash last columns
                    if columns_start != -1:
                        columns_index_one_table.append([columns_start, i])
                        columns_index.append(columns_index_one_table)

                    # if end
                    if i + 1 == length or x[i + 1] == 0:
                        break

                if x[i] == 140:
                    # stash the table
                    tables_index.append([i + 1, x.index(102, i + 1)])

                    # initialize for this table
                    columns_start = -1
                    columns_index_one_table = []

                # stash a column
                if x[i] == 102:
                    if columns_start != -1:
                        columns_index_one_table.append([columns_start, i])

                    if i + 1 < length and x[i + 1] != 0:
                        columns_start = i + 1

            # calculate the max-length table-column
            for i in range(len(tables_index)):
                for each in columns_index[i]:
                    if max_len < tables_index[i][1] - tables_index[i][0] + each[1] - each[0]:
                        max_len = tables_index[i][1] - tables_index[i][0] + each[1] - each[0]
                        max_id = cnt

            idx['tables'].append(tables_index)
            idx['columns'].append(columns_index)
            cnt += 1

        # show the max-length table-column
        print('max_len is', max_len)
        print('that is', query_with_db_info[max_id])

        # 5. store the result
        X = {
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask,
            'idx': idx,
            'query_with_db_info': query_with_db_info,
            'question_id': question_id,
            'db_name': db_name,
        }

        with open(store_link, 'w') as f:
            f.write(json.dumps(X, ensure_ascii=False, indent=4, separators=(',', ':')))

    @staticmethod
    def __generate_y(data: dict, content: dict, schema: dict, store_folder: str):
        """generate the ground truth for each sample in X.

        :param data: a dict contains db_name, question_id and question.
        :param content: content of database given by db_name in data.
        :param schema: schema of database given by db_name in data.
        :param store_folder: the location to store the result.
        :return: None
        """
        pass


if __name__ == '__main__':
    DataPreprocessor.preprocess(data_source='Train')
    DataPreprocessor.preprocess(data_source='Validation')
