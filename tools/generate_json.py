import json
import os

result_path = "/Users/cindy/others/Result/"
datapath = result_path+'data.json'
schema_path = result_path+'db_schema.json'
print(os.getcwd())
qd = "question_id"


def initDic(dataPath):
    idToDB = {}

    # 加载测试数据，得到其 db_name
    with open(dataPath, 'r', encoding='UTF-8') as f:
        test_data = json.load(f)
        for each in test_data:
            idToDB[each["question_id"]] = each["db_name"]

    return idToDB


def getContent(schema_path):
    dbToColumn = {}

    with open(schema_path, 'r', encoding='UTF-8') as f:
        schema = json.load(f)

    # dbToColumn["db_name"][col_idx] = [table_id,col_name]
    for each in schema:
        db = each['db_name']
        dbToColumn[db] = {}
        # pair ["化学_0", "化学id"]
        for idx, pair in enumerate(each['col_name'][1:]):
            dbToColumn[db][idx + 1] = [pair[0], pair[1]]

    return dbToColumn

def trans(val):
    if val == 0:
        return -999
    if val == 1:
        return -1
    return val-1


def index_trans(value, key):
    for i in range(len(value[key])):
        if type(value[key][i]) == int:
            value[key][i] = trans(value[key][i])
        else:
            for j in range(len(value[key][i])):
                value[key][i][j] = trans(value[key][i][j])
    return value


def rebuild(sql, source, key, mode=False):
    for i in range(len(source[qd])):
        if mode:
            sql[source[qd][i]][key] = source[key][i]
        else:
            sql[source[qd][i]][key].append(source[key][i])
    return sql


def select_insert(target, val, k, filt=-1):
    for i in range(len(val)):
        if filt != -1 and i > 0:
            break
        target[i][k] = val[i]
    return target


def gen_select():
    with open(result_path + 'Select/K', 'r') as f:
        select_k = json.load(f)
    with open(result_path + 'Select/prefix', 'r') as f:
        select_prefix = json.load(f)
    with open(result_path + 'Select/agg', 'r') as f:
        select_agg = json.load(f)
    with open(result_path + 'Select/com', 'r') as f:
        select_com = json.load(f)
    with open(result_path + 'Select/suffix', 'r') as f:
        select_suffix = json.load(f)
    select_prefix = index_trans(select_prefix, 'prefix')
    select_suffix = index_trans(select_suffix, 'suffix')

    sql = {}
    # select
    for i in select_k[qd]:
        sql[i] = {'K': [],'prefix': [], 'agg': [], 'com': [], 'suffix': []}
    sql = rebuild(sql, select_k, 'K', True)
    sql = rebuild(sql, select_prefix, 'prefix', True)
    sql = rebuild(sql, select_agg, 'agg')
    sql = rebuild(sql, select_com, 'com')
    sql = rebuild(sql, select_suffix, 'suffix')
    res = {}
    for k in sql.keys():
        num, prefix, agg, com, suffix = sql[k]['K'],sql[k]['prefix'], sql[k]['agg'], sql[k]['com'], sql[k]['suffix']
        if not num: continue
        select = [[0, 0, 0, 0] for _ in range(num)]
        select = select_insert(select, prefix, 0)
        select = select_insert(select, agg, 1)
        select = select_insert(select, com, 2, 1)
        select = select_insert(select, suffix, 3, 1)
        res[k] = select

    return res


def gen_orderBy():
    with open(result_path + 'OrderBy/col', 'r') as f:
        col = json.load(f)
    with open(result_path + 'OrderBy/order', 'r') as f:
        order = json.load(f)
    col = index_trans(col, 'col')
    sql = {}
    for i in range(len(order[qd])):
        sql[order[qd][i]] = {'col': [], 'order': order['order'][i]}
    for i in range(len(col[qd])):
        sql[col[qd][i]]['col'] = col['col'][i]
    res = {}
    for k in sql.keys():
        t1, t2 = sql[k]['order'], sql[k]['col']
        if not t1:
            res[k] = []
        else:
            s = "ASC" if t1 == 1 else "DESC"
            res[k] = [s, [[t2, 0, 0, 0]]]
    return res


def gen_limit():
    with open(result_path + 'Limit/need', 'r') as f:
        limit = json.load(f)
    res = {}
    for i in range(len(limit[qd])):
        res[limit[qd][i]] = limit['need'][i]
    return res


def gen_groupBy():
    with open(result_path + 'GroupBy/col', 'r') as f:
        col = json.load(f)
    with open(result_path + 'GroupBy/need', 'r') as f:
        need = json.load(f)
    col = index_trans(col, 'col')
    res = {}
    sql = {}
    for i in range(len(need[qd])):
        sql[need[qd][i]] = {'need': need['need'][i], 'col':[]}
    for i in range(len(col[qd])):
        sql[col[qd][i]]['col'].append(col['col'][i])

    # 有待商榷
    for k in sql.keys():
        if not sql[k]['need']:
            res[k] = []
        else:
            res[k] = sql[k]['col']
    return res


def gen_from(select_info):
    with open(result_path + 'From/J', 'r') as f:
        J = json.load(f)
    with open(result_path + 'From/N', 'r') as f:
        N = json.load(f)
    with open(result_path + 'From/prefix', 'r') as f:
        prefix = json.load(f)
    with open(result_path + 'From/suffix', 'r') as f:
        suffix = json.load(f)
    prefix = index_trans(prefix, 'prefix')
    suffix = index_trans(suffix, 'suffix')
    res = {}
    sql = {}

    for i in N[qd]:
        sql[i] = {'N': [],'prefix': [], 'suffix': []}
    sql = rebuild(sql, N, 'N', True)
    sql = rebuild(sql, prefix, 'prefix', True)
    sql = rebuild(sql, suffix, 'suffix')
    for k in sql.keys():
        num, prefix, suffix = sql[k]['N'], sql[k]['prefix'], sql[k]['suffix']
        if not num:
            res[k] = []
            continue
        select = [[0, 0, 2, 0, 0, 0,{}] for _ in range(num)]
        select = select_insert(select, prefix, 0)
        select = select_insert(select, suffix, 5)
        res[k] = {'conds': select, 'table_ids': []}

    idToDB = initDic(datapath)
    dbToCol = getContent(schema_path)
    for k in select_info.keys():
        db = idToDB[k]
        table_ids = set()
        for it in select_info[k]:
            if it[0]!=-1:
                table_id = dbToCol[db][it[0]]
                table_ids.add(table_id[0])
            if it[3]!=0:
                table_id = dbToCol[db][it[0]]
                table_ids.add(table_id[0])
        tmp = [['table_id', it] for it in table_ids]
        res[k]['table_ids'] = tmp

    return res


select = gen_select()
orderBy = gen_orderBy()
limit = gen_limit()
groupBy = gen_groupBy()
From = gen_from(select)
print("end")


