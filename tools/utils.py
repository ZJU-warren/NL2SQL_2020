import json

result_path = "../../Result/"
datapath = result_path+'data.json'
schema_path = result_path+'db_schema.json'
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


def trans(val, mode=''):
    if val == 0: return -999
    if val == 1 and mode == 'select':
        return -1
    return val-1


def index_trans(value, key, mode=''):
    for i in range(len(value[key])):
        if type(value[key][i]) == int:
            value[key][i] = trans(value[key][i], mode)
        else:
            for j in range(len(value[key][i])):
                value[key][i][j] = trans(value[key][i][j], mode)
    return value


def rebuild(sql, source, key, mode=False):
    # 遍历 source 所有的 question_id
    for i in range(len(source[qd])):
        if mode:
            sql[source[qd][i]][key] = source[key][i]
        else:
            sql[source[qd][i]][key].append(source[key][i])
    return sql


def select_insert(target, val, k, filt=-1):
    for i in range(len(val)):
        # 对于 suffix，agg，com
        # 只预测第一个，预测为0
        if filt != -1 and i > 0:
            break
        target[i][k] = val[i]
    return target
