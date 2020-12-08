import json
import os
from tools.utils import *


class JsonGenerator:
    def __init__(self, file_path, data_path, schema_path):
        self.file_path = file_path
        self.data_path = data_path
        self.schema_path = schema_path
        self.self.qid = "question_id"
        self.idToDB = initDic(datapath)
        self.dbToColumn = getContent(schema_path)

    def gen_select(self):
        with open(self.file_path + 'Select/K', 'r') as f:
            select_k = json.load(f)
        with open(self.file_path + 'Select/prefix', 'r') as f:
            select_prefix = json.load(f)
        with open(self.file_path + 'Select/agg', 'r') as f:
            select_agg = json.load(f)
        with open(self.file_path + 'Select/com', 'r') as f:
            select_com = json.load(f)
        with open(self.file_path + 'Select/suffix', 'r') as f:
            select_suffix = json.load(f)
        select_prefix = index_trans(select_prefix, 'prefix')
        select_suffix = index_trans(select_suffix, 'suffix')

        sql = {}
        # select
        for i in select_k[self.qid]:
            sql[i] = {'K': [], 'prefix': [], 'agg': [], 'com': [], 'suffix': []}
        sql = rebuild(sql, select_k, 'K', True)
        sql = rebuild(sql, select_prefix, 'prefix', True)
        sql = rebuild(sql, select_agg, 'agg')
        sql = rebuild(sql, select_com, 'com')
        sql = rebuild(sql, select_suffix, 'suffix')
        res = {}
        for k in sql.keys():
            num, prefix, agg, com, suffix = sql[k]['K'], sql[k]['prefix'], sql[k]['agg'], sql[k]['com'], sql[k][
                'suffix']
            if not num: continue
            select = [[0, 0, 0, 0] for _ in range(num)]
            select = select_insert(select, prefix, 0)
            select = select_insert(select, agg, 1)
            select = select_insert(select, com, 2, 1)
            select = select_insert(select, suffix, 3, 1)
            res[k] = select

        return res

    def gen_orderBy(self):
        with open(self.file_path + 'OrderBy/col', 'r') as f:
            col = json.load(f)
        with open(self.file_path + 'OrderBy/order', 'r') as f:
            order = json.load(f)
        col = index_trans(col, 'col')
        sql = {}
        for i in range(len(order[self.qid])):
            sql[order[self.qid][i]] = {'col': [], 'order': order['order'][i]}
        for i in range(len(col[self.qid])):
            sql[col[self.qid][i]]['col'] = col['col'][i]
        res = {}
        for k in sql.keys():
            t1, t2 = sql[k]['order'], sql[k]['col']
            if not t1:
                res[k] = []
            else:
                s = "ASC" if t1 == 1 else "DESC"
                res[k] = [s, [[t2, 0, 0, 0]]]
        return res

    def gen_groupBy(self):
        with open(self.file_path + 'GroupBy/col', 'r') as f:
            col = json.load(f)
        with open(self.file_path + 'GroupBy/need', 'r') as f:
            need = json.load(f)
        col = index_trans(col, 'col')
        res = {}
        sql = {}
        for i in range(len(need[self.qid])):
            sql[need[self.qid][i]] = {'need': need['need'][i], 'col': []}
        for i in range(len(col[self.qid])):
            sql[col[self.qid][i]]['col'].append(col['col'][i])

        # 有待商榷
        for k in sql.keys():
            if not sql[k]['need']:
                res[k] = []
            else:
                res[k] = sql[k]['col']
        return res

    def gen_from(self, select_info, where_info=None, having_info=None):
        with open(self.file_path + 'From/J', 'r') as f:
            J = json.load(f)
        with open(self.file_path + 'From/N', 'r') as f:
            N = json.load(f)
        with open(self.file_path + 'From/prefix', 'r') as f:
            prefix = json.load(f)
        with open(self.file_path + 'From/suffix', 'r') as f:
            suffix = json.load(f)
        prefix = index_trans(prefix, 'prefix')
        suffix = index_trans(suffix, 'suffix')
        res = {}
        sql = {}

        for i in N[self.qid]:
            sql[i] = {'N': [], 'prefix': [], 'suffix': []}
        sql = rebuild(sql, N, 'N', True)
        sql = rebuild(sql, prefix, 'prefix', True)
        sql = rebuild(sql, suffix, 'suffix')
        for k in sql.keys():
            num, prefix, suffix = sql[k]['N'], sql[k]['prefix'], sql[k]['suffix']
            if not num:
                res[k] = {}
                continue
            select = [[0, 0, 2, 0, 0, 0, {}] for _ in range(num)]
            select = select_insert(select, prefix, 0)
            select = select_insert(select, suffix, 5)
            res[k] = {'conds': select, 'table_ids': []}

        for k in select_info.keys():
            db = self.idToDB[k]
            columns = []
            table_ids = set()
            columns.extend([it[0] for it in select_info[k]])
            columns.extend([it[3] for it in select_info[k]])
            columns.extend([it[0] for it in where_info[k]])
            columns.extend([it[5] for it in where_info[k]])
            columns.extend([it[0] for it in having_info[k]])
            columns.extend([it[5] for it in having_info[k]])

            for i in columns:
                if i == -1 or i == 0 or i==-999:
                    continue
                try:
                    table_ids.add(self.dbToColumn[db][i])
                except:
                    print("error when finding from table id part, ", db, i)

            res[k]['table_ids'] = [['table_id', it] for it in table_ids]

        return res

    def gen_others(self):
        with open(self.file_path + 'Where/where_clause.json', 'r') as f:
            where = json.load(f)
        with open(self.file_path + 'Having/having_clause.json', 'r') as f:
            having = json.load(f)
        with open(self.file_path + 'Limit/limit_clause.json', 'r') as f:
            limit = json.load(f)
        return where, having, limit

    def merge_sql(self):
        select_json, order_json, group_json = self.gen_select(), self.gen_orderBy(), self.gen_groupBy()
        where_json, having_json, limit_json = self.gen_others()
        from_json = self.gen_from(select_json, where_json, having_json)
        sql = {}
        for k in select_json.keys():
            sql = {'select': select_json[k], 'from': from_json[k], 'where':where_json[k], 'groupBy': group_json[k],
                   'orderBy': order_json[k], 'having': having_json[k], 'limit': limit_json[k],
                   'except': {}, 'union': {}, 'intersect': {}}
            sql[k] = sql

        return sql


class OutPutModule:
    def __init__(self, file_path, data_path, schema_path, sub_file_path=None):
        self.file_path = file_path
        self.sub_file_path = sub_file_path
        self.data_path = data_path
        self.schema_path = schema_path
        self.idToDB = initDic(datapath)
        self.sql = JsonGenerator(file_path, data_path, schema_path)
        self.sub_sql = JsonGenerator(sub_file_path, data_path, schema_path)

    def load_combination(self):
        with open(self.file_path + 'Combination/comb', 'r') as f:
            comb = json.load(f)
        res = {}
        for i in range(len(comb['comb'])):
            res[comb['question_id'][i]] = comb['comb'][i]
        return res

    def start(self, mode=True):
        comb_json = self.load_combination()
        sql = self.sql.merge_sql()
        res = []
        if mode:
            sub_sql = self.sub_sql.merge_sql()
            for k in comb_json.keys():
                item = {'question_id': k, 'db_name': self.idToDB[k]}
                if comb_json[k] == 0:
                    item['sql'] = sql
                if comb_json[k] == 1:
                    sql['except'] = sub_sql
                if comb_json[k] == 2:
                    sql['union'] = sub_sql
                if comb_json[k] == 3:
                    sql['intersect'] = sub_sql
                item['sql'] = sql
                res.append(item)
        else:
            for k in comb_json.keys():
                res.append({'question_id': k, 'db_name': self.idToDB[k], 'sql': sql})
        with open('res.json', 'w') as f:
            json.dump(res, f)
        print("save answer success!")



