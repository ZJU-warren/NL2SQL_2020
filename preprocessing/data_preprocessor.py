"""
Preprocess the data source to (X, y) for 'Train' and 'Validation' or X for 'Test'
"""

import json
import DataLinkSet as DLSet
from transformers import BertTokenizer
from tools.file_manager import generate_new_folder


class DataPreprocessor:
    tokenizer = BertTokenizer.from_pretrained(DLSet.vocab_path)

    @staticmethod
    def col_map(col):
        if col == -999:
            target = 0
        elif col == -1:
            target = 1
        else:
            target = col + 1
        return target

    @staticmethod
    def preprocess(data_source: str, cheat_mode: bool = True):
        """preprocess the given data source.

        :param data_source: the data sources need preprocess.
        :param cheat_mode: whether the data source contains ground truth.
        :return: None
        """
        print('\n' + '*' * 20 + ' preprocess %s ' % data_source + '*' * 20)

        # load raw data
        with open(DLSet.raw_data_link % data_source, 'r') as f:
            data = json.load(f)
        with open(DLSet.raw_content_link % data_source, 'r') as f:
            content = json.load(f)
        with open(DLSet.raw_schema_link % data_source, 'r') as f:
            schema = json.load(f)
        
        # clear the folder and generate new one
        generate_new_folder(DLSet.main_folder_link % data_source)

        # generate X after tokenizer
        DataPreprocessor.__generate_X(data, content, schema, DLSet.X_link % data_source)

        # in cheat mode, the ground truth should be format as y
        if cheat_mode is False:
            generate_new_folder(DLSet.main_folder_link % data_source + '/Select')
            generate_new_folder(DLSet.main_folder_link % data_source + '/From')
            generate_new_folder(DLSet.main_folder_link % data_source + '/Where')
            generate_new_folder(DLSet.main_folder_link % data_source + '/GroupBy')
            generate_new_folder(DLSet.main_folder_link % data_source + '/Having')
            generate_new_folder(DLSet.main_folder_link % data_source + '/Limit')
            generate_new_folder(DLSet.main_folder_link % data_source + '/GroupBy')
            generate_new_folder(DLSet.main_folder_link % data_source + '/Combination')
            return

        # generate y
        DataPreprocessor.__generate_y(data, content, schema, DLSet.main_folder_link % data_source)

    @staticmethod
    def __generate_db_info(schema: list):
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
    def __generate_X(data: list, content: list, schema: list, store_link: str):
        """generate the X contains query, tables, columns information after tokenize.

        :param data: a list contains db_name, question_id and question.
        :param content: content of database given by db_name in data.
        :param schema: schema of database given by db_name in data.
        :param store_link: the location to store the result.
        :return: None
        """

        print('-------------------- generate X --------------------')

        # 1. get the toString information of each database
        db_info = DataPreprocessor.__generate_db_info(schema)

        # 2. combine query append with the corresponded database's information, and load others data
        query_with_db_info = []
        question_id = []
        db_name = []
        cnt = 0
        for each in data:
            temp = each['question'] + db_info[each['db_name']]
            if len(temp) > 505:
                temp = '[unused1]'.join(temp[:505].split('[unused1]')[: -1])
                cnt += 1
            query_with_db_info.append(temp)
            question_id.append(each['question_id'])
            db_name.append(each['db_name'])
        print('cnt = %d', cnt)
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

                if x[i] == 140 and x[i-1] == 8148:
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

            # print('tables_index', tables_index)
            # print('columns_index', columns_index)
            # print('len x = ',len(x), 'x =', x)
            #
            # print(len(tables_index),
            #       len(columns_index),
            #       len(query_with_db_info[cnt].split('[unused1]')),
            #       len(query_with_db_info[cnt].split('[SEP]'))
            #       )
            #
            # print(cnt, query_with_db_info[cnt])
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
        print(store_link)
        print('X is', len(X['input_ids']))
        with open(store_link, 'w') as f:
            f.write(json.dumps(X, ensure_ascii=False, indent=4, separators=(',', ':')))

    @staticmethod
    def __generate_y(data: list, content: list, schema: list, store_folder: str):
        """generate the ground truth for each sample in X.

        :param data: a list contains db_name, question_id and question.
        :param content: content of database given by db_name in data.
        :param schema: schema of database given by db_name in data.
        :param store_folder: the folder location to store the result.
        :return: None
        """

        print('-------------------- generate y --------------------')

        DataPreprocessor.__generate_y_select(data, content, schema, store_folder + '/Select')
        DataPreprocessor.__generate_y_from(data, content, schema, store_folder + '/From')
        DataPreprocessor.__generate_y_where(data, content, schema, store_folder + '/Where')
        DataPreprocessor.__generate_y_group_by(data, content, schema, store_folder + '/GroupBy')
        DataPreprocessor.__generate_y_having(data, content, schema, store_folder + '/Having')
        DataPreprocessor.__generate_y_limit(data, content, schema, store_folder + '/Limit')
        DataPreprocessor.__generate_y_order_by(data, content, schema, store_folder + '/OrderBy')
        DataPreprocessor.__generate_y_combination(data, content, schema, store_folder + '/Combination')

    @staticmethod
    def __filter(raw_data: list):
        result = []
        cnt = 0
        for each in raw_data:
            cnt += 1
            # if the selects columns index is 0
            flag = True
            # print(cnt, each)
            for _ in each['sql']['select']:
                if _[0] == 0:
                    flag = False

            for _ in each['sql']['from']['table_ids']:
                if _[0] == 'table_id' and _[1] == -1:
                    flag = False
                    break
            if flag:
                result.append(each)
            else:
                result.append(None)
        return result

    @staticmethod
    def __generate_y_select(raw_data: list, content: list, schema: list, store_folder: str):
        """generate the ground truth of select part for each sample in X.

        :param data: a list contains db_name, question_id and question.
        :param content: content of database given by db_name in data.
        :param schema: schema of database given by db_name in data.
        :param store_folder: the folder location to store the result.
        :return: None
        """

        def __generate_cols():
            """ conditions: the conditions """
            X_gt_sup_K = {'X_id': []}
            y_gt_K = {'K': []}

            X_gt_sup_prefix = {'X_id': [], 'K': []}
            y_gt_prefix = {'prefix': []}

            X_gt_sup_com = {'X_id': [], 'prefix': []}
            y_gt_com = {'com': []}

            X_gt_sup_agg = {'X_id': [], 'prefix': []}
            y_gt_agg = {'agg': []}

            X_gt_sup_suffix = {'X_id': [], 'prefix': []}
            y_gt_suffix = {'suffix': []}

            for i in range(total):
                if data[i] is None:
                    continue

                X_gt_sup_K['X_id'].append(i)
                y_gt_K['K'].append(len(data[i]['sql']['select']))

                # prefix_col
                targets = []
                for _ in data[i]['sql']['select']:
                    targets.append(DataPreprocessor.col_map(_[0]))
                X_gt_sup_prefix['X_id'].append(i)
                X_gt_sup_prefix['K'].append(len(data[i]['sql']['select']))
                y_gt_prefix['prefix'].append(targets)

                # agg
                for _ in data[i]['sql']['select']:
                    X_gt_sup_agg['X_id'].append(i)
                    X_gt_sup_agg['prefix'].append(DataPreprocessor.col_map(_[0]))
                    y_gt_agg['agg'].append(_[1])

                # com
                for _ in data[i]['sql']['select']:
                    X_gt_sup_com['X_id'].append(i)
                    X_gt_sup_com['prefix'].append(DataPreprocessor.col_map(_[0]))
                    y_gt_com['com'].append(_[2])

                # suffix_col
                for _ in data[i]['sql']['select']:
                    X_gt_sup_suffix['X_id'].append(i)
                    X_gt_sup_suffix['prefix'].append(DataPreprocessor.col_map(_[0]))
                    y_gt_suffix['suffix'].append(_[3])

            with open(store_folder + '/X_gt_sup_K', 'w') as f:
                f.write(json.dumps(X_gt_sup_K, ensure_ascii=False, indent=4, separators=(',', ':')))
            with open(store_folder + '/y_gt_K', 'w') as f:
                f.write(json.dumps(y_gt_K, ensure_ascii=False, indent=4, separators=(',', ':')))

            with open(store_folder + '/X_gt_sup_prefix', 'w') as f:
                f.write(json.dumps(X_gt_sup_prefix, ensure_ascii=False, indent=4, separators=(',', ':')))
            with open(store_folder + '/y_gt_prefix', 'w') as f:
                f.write(json.dumps(y_gt_prefix, ensure_ascii=False, indent=4, separators=(',', ':')))

            with open(store_folder + '/X_gt_sup_com', 'w') as f:
                f.write(json.dumps(X_gt_sup_com, ensure_ascii=False, indent=4, separators=(',', ':')))
            with open(store_folder + '/y_gt_com', 'w') as f:
                f.write(json.dumps(y_gt_com, ensure_ascii=False, indent=4, separators=(',', ':')))

            with open(store_folder + '/X_gt_sup_agg', 'w') as f:
                f.write(json.dumps(X_gt_sup_agg, ensure_ascii=False, indent=4, separators=(',', ':')))
            with open(store_folder + '/y_gt_agg', 'w') as f:
                f.write(json.dumps(y_gt_agg, ensure_ascii=False, indent=4, separators=(',', ':')))

            with open(store_folder + '/X_gt_sup_suffix', 'w') as f:
                f.write(json.dumps(X_gt_sup_suffix, ensure_ascii=False, indent=4, separators=(',', ':')))
            with open(store_folder + '/y_gt_suffix', 'w') as f:
                f.write(json.dumps(y_gt_suffix, ensure_ascii=False, indent=4, separators=(',', ':')))

        # 0.
        print('-------------------- generate select part--------------------')
        generate_new_folder(store_folder)

        # 1. filter the wrong data
        data = DataPreprocessor.__filter(raw_data)
        total = len(data)

        # 2. the value of columns
        __generate_cols()

    @staticmethod
    def __generate_y_from(raw_data: list, content: list, schema: list, store_folder: str):
        """generate the ground truth of from part for each sample in X.

        :param raw_data: a list contains db_name, question_id and question.
        :param content: content of database given by db_name in data.
        :param schema: schema of database given by db_name in data.
        :param store_folder: the folder location to store the result.
        :return: None
        """
        def __generate_conditions():
            """ conditions: the conditions """
            X_gt_sup_N = {'X_id': []}
            y_gt_N = {'N': []}

            X_gt_sup_prefix = {'X_id': [], 'N': []}
            X_gt_sup_suffix = {'X_id': [], 'prefix': []}

            y_gt_prefix = {'prefix': []}
            y_gt_suffix = {'suffix': []}

            for i in range(total):
                if data[i] is None:
                    continue

                n = min(2, len(data[i]['sql']['from']['conds']))
                X_gt_sup_N['X_id'].append(i)
                y_gt_N['N'].append(n)

                if n == 0:
                    continue

                # prefix and suffix
                X_gt_sup_prefix['X_id'].append(i)
                X_gt_sup_prefix['N'].append(n)

                targets = []
                for _ in data[i]['sql']['from']['conds']:
                    if type(_) == list:
                        prefix = DataPreprocessor.col_map(_[0])
                        targets.append(prefix)

                        X_gt_sup_suffix['X_id'].append(i)
                        X_gt_sup_suffix['prefix'].append(prefix)
                        y_gt_suffix['suffix'].append(DataPreprocessor.col_map(_[5]))

                y_gt_prefix['prefix'].append(targets)

            with open(store_folder + '/X_gt_sup_N', 'w') as f:
                f.write(json.dumps(X_gt_sup_N, ensure_ascii=False, indent=4, separators=(',', ':')))
            with open(store_folder + '/y_gt_N', 'w') as f:
                f.write(json.dumps(y_gt_N, ensure_ascii=False, indent=4, separators=(',', ':')))

            with open(store_folder + '/X_gt_sup_prefix', 'w') as f:
                f.write(json.dumps(X_gt_sup_prefix, ensure_ascii=False, indent=4, separators=(',', ':')))
            with open(store_folder + '/y_gt_prefix', 'w') as f:
                f.write(json.dumps(y_gt_prefix, ensure_ascii=False, indent=4, separators=(',', ':')))

            with open(store_folder + '/X_gt_sup_suffix', 'w') as f:
                f.write(json.dumps(X_gt_sup_suffix, ensure_ascii=False, indent=4, separators=(',', ':')))
            with open(store_folder + '/y_gt_suffix', 'w') as f:
                f.write(json.dumps(y_gt_suffix, ensure_ascii=False, indent=4, separators=(',', ':')))

        def __generate_tables_sqls():
            """ J: the amount of tables """
            X_gt_sup_J = {'X_id': []}
            y_gt_J = {'J': []}

            X_gt_sup_tables = {'X_id': [], 'J': []}
            y_gt_tables = {'tables': []}

            X_gt_sup_sqls = {'X_id': []}
            y_gt_sqls = {'sqls': []}

            for i in range(total):
                if data[i] is None:
                    continue

                J = 0
                target_tables = []
                target_sqls = []
                for _ in data[i]['sql']['from']['table_ids']:
                    if _[0] == 'table_id':
                        J += 1
                        target_tables.append(_[1])
                    else:
                        target_sqls.append(_[1])

                # J
                X_gt_sup_J['X_id'].append(i)
                y_gt_J['J'].append(J)

                # tables
                if J != 0:
                    X_gt_sup_tables['X_id'].append(i)
                    y_gt_tables['tables'].append(target_tables)

                # sqls
                else:
                    X_gt_sup_sqls['X_id'].append(i)
                    temp = data[i].copy()
                    temp['sql'] = target_sqls
                    y_gt_sqls['sqls'].append(temp)
                    # y_gt_sqls['sqls'].append(target_sqls)

                    # print('num of sqls:', len(data[i]['sql']['from']['table_ids']))
                    # print('\t select:', data[i]['sql']['select'])
                    # print('\t sql1:', data[i]['sql']['from']['table_ids'][0][1])
                    # print('\t sql2:', data[i]['sql']['from']['table_ids'][1][1])

            with open(store_folder + '/X_gt_sup_J', 'w') as f:
                f.write(json.dumps(X_gt_sup_J, ensure_ascii=False, indent=4, separators=(',', ':')))
            with open(store_folder + '/y_gt_J', 'w') as f:
                f.write(json.dumps(y_gt_J, ensure_ascii=False, indent=4, separators=(',', ':')))
            with open(store_folder + '/X_gt_sup_tables', 'w') as f:
                f.write(json.dumps(X_gt_sup_tables, ensure_ascii=False, indent=4, separators=(',', ':')))
            with open(store_folder + '/y_gt_tables', 'w') as f:
                f.write(json.dumps(y_gt_tables, ensure_ascii=False, indent=4, separators=(',', ':')))
            with open(store_folder + '/X_gt_sup_sqls', 'w') as f:
                f.write(json.dumps(X_gt_sup_sqls, ensure_ascii=False, indent=4, separators=(',', ':')))
            with open(store_folder + '/y_gt_sqls', 'w') as f:
                f.write(json.dumps(y_gt_sqls, ensure_ascii=False, indent=4, separators=(',', ':')))

        # 0.
        print('-------------------- generate from part--------------------')
        generate_new_folder(store_folder)

        # 1. filter the wrong data
        data = DataPreprocessor.__filter(raw_data)
        total = len(data)

        # 2. the conditions
        __generate_conditions()

        # 3. the tables and sub-sqls
        __generate_tables_sqls()

    @staticmethod
    def __generate_y_where(raw_data: list, content: list, schema: list, store_folder: str):
        """generate the ground truth of where part for each sample in X.

        :param raw_data: a list contains db_name, question_id and question.
        :param content: content of database given by db_name in data.
        :param schema: schema of database given by db_name in data.
        :param store_folder: the folder location to store the result.
        :return: None
        """
        def __generate_conditions():
            """ conditions: the conditions """
            X_gt_sup_N = {'X_id': []}
            y_gt_N = {'N': []}

            X_gt_sup_operation = {'X_id': []}
            y_gt_operation = {'operation': []}

            X_gt_sup_prefix = {'X_id': [], 'N': []}
            y_gt_prefix = {'prefix': []}

            X_gt_sup_com = {'X_id': [], 'prefix': []}
            y_gt_com = {'com': []}

            # X_gt_sup_value = {'X_id': [], 'prefix': [], 'com': []}
            # y_gt_value = {'value': []}

            X_gt_sup_eq = {'X_id': [], 'prefix': [], 'com': [], 'value': []}
            y_gt_eq = {'eq': []}

            X_gt_sup_suffix = {'X_id': [], 'prefix': [], 'com': []}
            y_gt_suffix = {'suffix': []}

            X_gt_sup_sql = {'X_id': []}
            y_gt_sql = {'sql': []}

            for i in range(total):
                if data[i] is None:
                    continue

                # N
                N = min(2, len(data[i]['sql']['where']))
                X_gt_sup_N['X_id'].append(i)
                y_gt_N['N'].append(N)

                if N == 0:
                    continue

                # operation
                if N == 2:
                    X_gt_sup_operation['X_id'].append(i)
                    y_gt_operation['operation'].append(0 if data[i]['sql']['where'][1] == 'AND' else 1)

                # prefix, com, value and suffix
                targets = []
                for _ in data[i]['sql']['where']:
                    if type(_) == list:
                        # prefix
                        prefix = DataPreprocessor.col_map(_[0])
                        targets.append(prefix)

                        # agg is default 0

                        # eq
                        if True:
                            X_gt_sup_eq['X_id'].append(i)
                            X_gt_sup_eq['prefix'].append(prefix)
                            y_gt_eq['eq'].append(_[2])

                        # # value
                        # # ____________ filter by com ________________
                        # if True:
                        #     X_gt_sup_value['X_id'].append(i)
                        #     X_gt_sup_value['prefix'].append(_[0])
                        #     X_gt_sup_value['com'].append(_[2])
                        #     y_gt_value['value'].append(_[3])

                        # com
                        X_gt_sup_com['X_id'].append(i)
                        X_gt_sup_com['prefix'].append(prefix)
                        y_gt_com['com'].append(_[4])

                        # suffix or sql
                        # ____________ filter by com ________________
                        if _[4] != 0:
                            # ____________ filter by eq ________________
                            if _[2] < 8:
                                X_gt_sup_suffix['X_id'].append(i)
                                X_gt_sup_suffix['prefix'].append(prefix)
                                y_gt_suffix['suffix'].append(DataPreprocessor.col_map(_[5]))
                            else:
                                X_gt_sup_sql['X_id'].append(i)
                                temp = data[i].copy()
                                temp['sql'] = _[6]
                                y_gt_sql['sql'].append(temp)
                                # y_gt_sql['sql'].append(_[6])

                X_gt_sup_prefix['X_id'].append(i)
                X_gt_sup_prefix['N'].append(N)
                y_gt_prefix['prefix'].append(targets)

            with open(store_folder + '/X_gt_sup_N', 'w') as f:
                f.write(json.dumps(X_gt_sup_N, ensure_ascii=False, indent=4, separators=(',', ':')))
            with open(store_folder + '/y_gt_N', 'w') as f:
                f.write(json.dumps(y_gt_N, ensure_ascii=False, indent=4, separators=(',', ':')))

            with open(store_folder + '/X_gt_sup_operation', 'w') as f:
                f.write(json.dumps(X_gt_sup_operation, ensure_ascii=False, indent=4, separators=(',', ':')))
            with open(store_folder + '/y_gt_operation', 'w') as f:
                f.write(json.dumps(y_gt_operation, ensure_ascii=False, indent=4, separators=(',', ':')))

            with open(store_folder + '/X_gt_sup_prefix', 'w') as f:
                f.write(json.dumps(X_gt_sup_prefix, ensure_ascii=False, indent=4, separators=(',', ':')))
            with open(store_folder + '/y_gt_prefix', 'w') as f:
                f.write(json.dumps(y_gt_prefix, ensure_ascii=False, indent=4, separators=(',', ':')))

            with open(store_folder + '/X_gt_sup_com', 'w') as f:
                f.write(json.dumps(X_gt_sup_com, ensure_ascii=False, indent=4, separators=(',', ':')))
            with open(store_folder + '/y_gt_com', 'w') as f:
                f.write(json.dumps(y_gt_com, ensure_ascii=False, indent=4, separators=(',', ':')))

            with open(store_folder + '/X_gt_sup_eq', 'w') as f:
                f.write(json.dumps(X_gt_sup_eq, ensure_ascii=False, indent=4, separators=(',', ':')))
            with open(store_folder + '/y_gt_eq', 'w') as f:
                f.write(json.dumps(y_gt_eq, ensure_ascii=False, indent=4, separators=(',', ':')))

            with open(store_folder + '/X_gt_sup_suffix', 'w') as f:
                f.write(json.dumps(X_gt_sup_suffix, ensure_ascii=False, indent=4, separators=(',', ':')))
            with open(store_folder + '/y_gt_suffix', 'w') as f:
                f.write(json.dumps(y_gt_suffix, ensure_ascii=False, indent=4, separators=(',', ':')))

            with open(store_folder + '/X_gt_sup_sql', 'w') as f:
                f.write(json.dumps(X_gt_sup_sql, ensure_ascii=False, indent=4, separators=(',', ':')))
            with open(store_folder + '/y_gt_sql', 'w') as f:
                f.write(json.dumps(y_gt_sql, ensure_ascii=False, indent=4, separators=(',', ':')))

        # 0.
        print('-------------------- generate where part--------------------')
        generate_new_folder(store_folder)

        # 1. filter the wrong data
        data = DataPreprocessor.__filter(raw_data)
        total = len(data)

        # 2. conditions
        __generate_conditions()

    @staticmethod
    def __generate_y_group_by(raw_data: list, content: list, schema: list, store_folder: str):
        """generate the ground truth of group by part for each sample in X.

        :param raw_data: a list contains db_name, question_id and question.
        :param content: content of database given by db_name in data.
        :param schema: schema of database given by db_name in data.
        :param store_folder: the folder location to store the result.
        :return: None
        """

        def __generate_group_by():
            X_gt_sup_need = {'X_id': []}
            y_gt_need = {'need': []}

            X_gt_sup_col = {'X_id': []}
            y_gt_col = {'col': []}

            for i in range(total):
                if data[i] is None:
                    continue

                X_gt_sup_need['X_id'].append(i)
                y_gt_need['need'].append(len(data[i]['sql']['groupBy']))

                if data[i]['sql']['groupBy']:
                    X_gt_sup_col['X_id'].append(i)
                    y_gt_col['col'].append(DataPreprocessor.col_map(data[i]['sql']['groupBy'][0]))

            with open(store_folder + '/X_gt_sup_need', 'w') as f:
                f.write(json.dumps(X_gt_sup_need, ensure_ascii=False, indent=4, separators=(',', ':')))
            with open(store_folder + '/y_gt_need', 'w') as f:
                f.write(json.dumps(y_gt_need, ensure_ascii=False, indent=4, separators=(',', ':')))
            with open(store_folder + '/X_gt_sup_col', 'w') as f:
                f.write(json.dumps(X_gt_sup_col, ensure_ascii=False, indent=4, separators=(',', ':')))
            with open(store_folder + '/y_gt_col', 'w') as f:
                f.write(json.dumps(y_gt_col, ensure_ascii=False, indent=4, separators=(',', ':')))
        # 0.
        print('-------------------- generate group by part--------------------')
        generate_new_folder(store_folder)

        # 1. filter the wrong data
        data = DataPreprocessor.__filter(raw_data)
        total = len(data)

        # 2. group_by
        __generate_group_by()

    @staticmethod
    def __generate_y_having(raw_data: list, content: list, schema: list, store_folder: str):
        """generate the ground truth of having part for each sample in X.

        :param raw_data: a list contains db_name, question_id and question.
        :param content: content of database given by db_name in data.
        :param schema: schema of database given by db_name in data.
        :param store_folder: the folder location to store the result.
        :return: None
        """

        def __generate_conditions():
            """ conditions: the conditions """
            X_gt_sup_N = {'X_id': []}
            y_gt_N = {'N': []}

            X_gt_sup_operation = {'X_id': []}
            y_gt_operation = {'operation': []}

            X_gt_sup_prefix = {'X_id': [], 'N': []}
            y_gt_prefix = {'prefix': []}

            X_gt_sup_agg = {'X_id': [], 'prefix': []}
            y_gt_agg = {'agg': []}

            X_gt_sup_eq = {'X_id': [], 'prefix': [], 'com': [], 'value': []}
            y_gt_eq = {'eq': []}

            # X_gt_sup_value = {'X_id': [], 'prefix': [], 'com': []}
            # y_gt_value = {'value': []}

            X_gt_sup_com = {'X_id': [], 'prefix': []}
            y_gt_com = {'com': []}

            X_gt_sup_suffix = {'X_id': [], 'prefix': [], 'com': []}
            y_gt_suffix = {'suffix': []}

            X_gt_sup_sql = {'X_id': []}
            y_gt_sql = {'sql': []}

            for i in range(total):
                if data[i] is None:
                    continue

                # N
                N = min(2, len(data[i]['sql']['having']))
                X_gt_sup_N['X_id'].append(i)
                y_gt_N['N'].append(N)

                if N == 0:
                    continue

                # operation
                if N == 2:
                    X_gt_sup_operation['X_id'].append(i)
                    y_gt_operation['operation'].append(0 if data[i]['sql']['having'][1] == 'AND' else 1)

                # prefix, com, value and suffix
                targets = []
                for _ in data[i]['sql']['having']:
                    if type(_) == list:
                        # prefix
                        prefix = DataPreprocessor.col_map(_[0])
                        targets.append(prefix)

                        # agg
                        X_gt_sup_agg['X_id'].append(i)
                        X_gt_sup_agg['prefix'].append(prefix)
                        y_gt_agg['agg'].append(_[1])

                        # eq
                        if True:
                            X_gt_sup_eq['X_id'].append(i)
                            X_gt_sup_eq['prefix'].append(prefix)
                            y_gt_eq['eq'].append(_[2])

                        # # value
                        # # ____________ filter by com ________________
                        # if True:
                        #     X_gt_sup_value['X_id'].append(i)
                        #     X_gt_sup_value['prefix'].append(_[0])
                        #     X_gt_sup_value['com'].append(_[2])
                        #     y_gt_value['value'].append(_[3])

                        # com
                        X_gt_sup_com['X_id'].append(i)
                        X_gt_sup_com['prefix'].append(prefix)
                        y_gt_com['com'].append(_[4])

                        # suffix or sql
                        # ____________ filter by com ________________
                        if _[4] != 0:
                            # ____________ filter by eq ________________
                            if _[2] < 8:
                                X_gt_sup_suffix['X_id'].append(i)
                                X_gt_sup_suffix['prefix'].append(prefix)
                                y_gt_suffix['suffix'].append(DataPreprocessor.col_map(_[5]))

                            else:
                                X_gt_sup_sql['X_id'].append(i)

                                temp = data[i].copy()
                                temp['sql'] = _[6]
                                y_gt_sql['sql'].append(temp)
                                # y_gt_sql['sql'].append(temp['sql'])

                X_gt_sup_prefix['X_id'].append(i)
                X_gt_sup_prefix['N'].append(N)
                y_gt_prefix['prefix'].append(targets)

            with open(store_folder + '/X_gt_sup_N', 'w') as f:
                f.write(json.dumps(X_gt_sup_N, ensure_ascii=False, indent=4, separators=(',', ':')))
            with open(store_folder + '/y_gt_N', 'w') as f:
                f.write(json.dumps(y_gt_N, ensure_ascii=False, indent=4, separators=(',', ':')))

            with open(store_folder + '/X_gt_sup_operation', 'w') as f:
                f.write(json.dumps(X_gt_sup_operation, ensure_ascii=False, indent=4, separators=(',', ':')))
            with open(store_folder + '/y_gt_operation', 'w') as f:
                f.write(json.dumps(y_gt_operation, ensure_ascii=False, indent=4, separators=(',', ':')))

            with open(store_folder + '/X_gt_sup_agg', 'w') as f:
                f.write(json.dumps(X_gt_sup_agg, ensure_ascii=False, indent=4, separators=(',', ':')))
            with open(store_folder + '/y_gt_agg', 'w') as f:
                f.write(json.dumps(y_gt_agg, ensure_ascii=False, indent=4, separators=(',', ':')))

            with open(store_folder + '/X_gt_sup_prefix', 'w') as f:
                f.write(json.dumps(X_gt_sup_prefix, ensure_ascii=False, indent=4, separators=(',', ':')))
            with open(store_folder + '/y_gt_prefix', 'w') as f:
                f.write(json.dumps(y_gt_prefix, ensure_ascii=False, indent=4, separators=(',', ':')))

            with open(store_folder + '/X_gt_sup_com', 'w') as f:
                f.write(json.dumps(X_gt_sup_com, ensure_ascii=False, indent=4, separators=(',', ':')))
            with open(store_folder + '/y_gt_com', 'w') as f:
                f.write(json.dumps(y_gt_com, ensure_ascii=False, indent=4, separators=(',', ':')))

            with open(store_folder + '/X_gt_sup_eq', 'w') as f:
                f.write(json.dumps(X_gt_sup_eq, ensure_ascii=False, indent=4, separators=(',', ':')))
            with open(store_folder + '/y_gt_eq', 'w') as f:
                f.write(json.dumps(y_gt_eq, ensure_ascii=False, indent=4, separators=(',', ':')))

            with open(store_folder + '/X_gt_sup_suffix', 'w') as f:
                f.write(json.dumps(X_gt_sup_suffix, ensure_ascii=False, indent=4, separators=(',', ':')))
            with open(store_folder + '/y_gt_suffix', 'w') as f:
                f.write(json.dumps(y_gt_suffix, ensure_ascii=False, indent=4, separators=(',', ':')))

            with open(store_folder + '/X_gt_sup_sql', 'w') as f:
                f.write(json.dumps(X_gt_sup_sql, ensure_ascii=False, indent=4, separators=(',', ':')))
            with open(store_folder + '/y_gt_sql', 'w') as f:
                f.write(json.dumps(y_gt_sql, ensure_ascii=False, indent=4, separators=(',', ':')))

        # 0.
        print('-------------------- generate having part--------------------')
        generate_new_folder(store_folder)

        # 1. filter the wrong data
        data = DataPreprocessor.__filter(raw_data)
        total = len(data)

        # 2. conditions
        __generate_conditions()

    @staticmethod
    def __generate_y_limit(raw_data: list, content: list, schema: list, store_folder: str):
        """generate the ground truth of limit part for each sample in X.

        :param raw_data: a list contains db_name, question_id and question.
        :param content: content of database given by db_name in data.
        :param schema: schema of database given by db_name in data.
        :param store_folder: the folder location to store the result.
        :return: None
        """

        # 0.
        print('-------------------- generate limit part--------------------')
        generate_new_folder(store_folder)

        def __generate_limit():
            X_gt_sup_need = {'X_id': []}
            y_gt_need = {'need': []}

            for i in range(total):
                if data[i] is None:
                    continue

                need = 1 if type(data[i]['sql']['limit']) is int  else 0
                X_gt_sup_need['X_id'].append(i)
                y_gt_need['need'].append(need)

            with open(store_folder + '/X_gt_sup_need', 'w') as f:
                f.write(json.dumps(X_gt_sup_need, ensure_ascii=False, indent=4, separators=(',', ':')))
            with open(store_folder + '/y_gt_need', 'w') as f:
                f.write(json.dumps(y_gt_need, ensure_ascii=False, indent=4, separators=(',', ':')))

        # 1. filter the wrong data
        data = DataPreprocessor.__filter(raw_data)
        total = len(data)

        # 2. limit
        __generate_limit()

    @staticmethod
    def __generate_y_order_by(raw_data: list, content: list, schema: list, store_folder: str):
        """generate the ground truth of order by part for each sample in X.

        :param raw_data: a list contains db_name, question_id and question.
        :param content: content of database given by db_name in data.
        :param schema: schema of database given by db_name in data.
        :param store_folder: the folder location to store the result.
        :return: None
        """

        def __generate_order_by():
            X_gt_sup_order = {'X_id': []}
            y_gt_order = {'order': []}

            X_gt_sup_col = {'X_id': []}
            y_gt_col = {'col': []}

            for i in range(total):
                if data[i] is None:
                    continue

                X_gt_sup_order['X_id'].append(i)
                if len(data[i]['sql']['orderBy']):
                    y_gt_order['order'].append(1 if data[i]['sql']['orderBy'][0] != 'DESC' else 2)

                    X_gt_sup_col['X_id'].append(i)
                    y_gt_col['col'].append(DataPreprocessor.col_map(data[i]['sql']['orderBy'][1][0][0]))
                else:
                    y_gt_order['order'].append(0)

            with open(store_folder + '/X_gt_sup_order', 'w') as f:
                f.write(json.dumps(X_gt_sup_order, ensure_ascii=False, indent=4, separators=(',', ':')))
            with open(store_folder + '/y_gt_order', 'w') as f:
                f.write(json.dumps(y_gt_order, ensure_ascii=False, indent=4, separators=(',', ':')))

            with open(store_folder + '/X_gt_sup_col', 'w') as f:
                f.write(json.dumps(X_gt_sup_col, ensure_ascii=False, indent=4, separators=(',', ':')))
            with open(store_folder + '/y_gt_col', 'w') as f:
                f.write(json.dumps(y_gt_col, ensure_ascii=False, indent=4, separators=(',', ':')))

        # 0.
        print('-------------------- generate order by part--------------------')
        generate_new_folder(store_folder)

        # 1. filter the wrong data
        data = DataPreprocessor.__filter(raw_data)
        total = len(data)

        # 2. group_by
        __generate_order_by()

    @staticmethod
    def __generate_y_combination(raw_data: list, content: list, schema: list, store_folder: str):
        """generate the ground truth of order by part for each sample in X.

        :param raw_data: a list contains db_name, question_id and question.
        :param content: content of database given by db_name in data.
        :param schema: schema of database given by db_name in data.
        :param store_folder: the folder location to store the result.
        :return: None
        """

        # 0.
        print('-------------------- generate combination by part--------------------')
        generate_new_folder(store_folder)

        def __generate_combination():
            X_gt_sup_comb = {'X_id': []}
            y_gt_comb = {'comb': []}

            X_gt_sup_sql = {'X_id': []}
            y_gt_sql = {'sql': []}

            for i in range(total):
                if data[i] is None:
                    continue

                if len(data[i]['sql']['except']):
                    comb = 1
                    X_gt_sup_sql['X_id'].append(i)
                    temp = data[i].copy()
                    temp['sql'] = data[i]['sql']['except']
                    y_gt_sql['sql'].append(temp)

                elif len(data[i]['sql']['union']):
                    comb = 2
                    X_gt_sup_sql['X_id'].append(i)
                    temp = data[i].copy()
                    temp['sql'] = data[i]['sql']['union']
                    y_gt_sql['sql'].append(temp)

                elif len(data[i]['sql']['intersect']):
                    comb = 3
                    X_gt_sup_sql['X_id'].append(i)
                    temp = data[i].copy()
                    temp['sql'] = data[i]['sql']['intersect']
                    y_gt_sql['sql'].append(temp)

                else:
                    comb = 0

                X_gt_sup_comb['X_id'].append(i)
                y_gt_comb['comb'].append(comb)

            with open(store_folder + '/X_gt_sup_comb', 'w') as f:
                f.write(json.dumps(X_gt_sup_comb, ensure_ascii=False, indent=4, separators=(',', ':')))
            with open(store_folder + '/y_gt_comb', 'w') as f:
                f.write(json.dumps(y_gt_comb, ensure_ascii=False, indent=4, separators=(',', ':')))

            with open(store_folder + '/X_gt_sup_sql', 'w') as f:
                f.write(json.dumps(X_gt_sup_sql, ensure_ascii=False, indent=4, separators=(',', ':')))
            with open(store_folder + '/y_gt_sql', 'w') as f:
                f.write(json.dumps(y_gt_sql, ensure_ascii=False, indent=4, separators=(',', ':')))

        # 1. filter the wrong data
        data = DataPreprocessor.__filter(raw_data)
        total = len(data)

        # 2. combination
        __generate_combination()
