import json

result_path = "../DataSet/Result/"

k = ['X_id', 'question', 'K', 'prefix']
with open(result_path+'Select/K', 'r') as f:
    select_k = json.load(f)
with open(result_path+'Select/prefix', 'r') as f:
    select_prefix = json.load(f)
l = len(select_prefix['question_id'])

res = []
for i in range(l):
    num = select_k[k[2]][i] + 1
    prefix = select_prefix['prefix'][i]
    data = {}
    select = []
    if len(prefix) != num:
        print("the question: %s has different length" % select_k['question_id'][i])
        continue
    for j in range(num):
        tmp = [prefix[j], 0, 0, 0]
        select.append(tmp)
    data['question'] = select_k['question_id'][i]
    data['sql'] = {'select': select}
    res.append(data)

print(res)
