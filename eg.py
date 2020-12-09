import json
import numpy as np
with open('../db/db_schema.json', 'r', encoding='utf-8') as f:
    db = json.load(f)
x = []
for it in db:
    table = it["table_name"]
    col_name = it['col_name']
    y = 0
    for i in range(1, len(col_name)):
        j = int(col_name[i][0].split('_')[-1])
        y += len(table[j]+col_name[i][1])
    if y>400: print(it['db_name'])
    x.append(y)
import pandas as pd

d = pd.DataFrame(x)
print(d.describe())
x = np.array(x)
c = (x>400)
print(np.sum(c))
