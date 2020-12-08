import json
import os

result_path = "/Users/cindy/others/Result/"
print(os.getcwd())
k = ['X_id', 'question', 'K', 'prefix']


def load_select():
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

    with open(result_path + 'Select/suffix', 'r') as f:
        select_suffix = json.load(f)
    with open(result_path + 'Select/suffix', 'r') as f:
        select_suffix = json.load(f)
    with open(result_path + 'Select/suffix', 'r') as f:
        select_suffix = json.load(f)
    with open(result_path + 'Select/suffix', 'r') as f:
        select_suffix = json.load(f)

    return select_k, select_prefix, select_agg, select_com, select_suffix


def load_from():
    with open(result_path + 'From/N', 'r') as f:
        from_n = json.load(f)
    with open(result_path + 'From/prefix', 'r') as f:
        from_prefix = json.load(f)
    with open(result_path + 'From/suffix', 'r') as f:
        from_suffix = json.load(f)
    return from_n, from_prefix, from_suffix


def gwn_json():
    select_k, select_prefix, select_agg, select_com, select_suffix = load_select()

    from_n, from_prefix, from_suffix = load_from()
    l = len(select_prefix['question_id'])
    res = []
    for i in range(l):

        # select
        select_num = select_k['K'][i] + 1
        prefix = select_prefix['prefix'][i]
        agg = select_agg['agg'][i]
        com = select_com['com'][i]
        suffix = select_suffix['suffix'][i]
        data = {}
        select = []
        for j in range(select_num):
            tmp = [prefix[j], agg, com, suffix]
            select.append(tmp)
        data['question'] = select_k['question_id'][i]

        res.append(data)

        # from


        # where

        # group by

        # order by

        # having

        # limit

        # set

    return res


res = gwn_json()
for i in res:
    print(i)


