import os 
import json 
import ipdb 
from lvis import LVIS, LVISVis

data_type = 'train'
train_path = f'/data1/niehui/lvis/annotations/lvis_v1_{data_type}_remove_rare.json'
data_type = 'val'
val_path = f'/data1/niehui/lvis/annotations/lvis_v1_{data_type}_remove_rare_855_categories.json'

with open(val_path) as fv, open(train_path) as ft:
    data_val = json.load(fv)
    data_train = json.load(ft)

# set data_val['categories'] as dasta_train['categories']

print(len(data_train['categories']))
print(len(data_val['categories']))
data_val['categories'] = data_train['categories']


# save data_val
new_anno_path = f'/data1/niehui/lvis/annotations/lvis_v1_{data_type}_remove_rare_866_categories.json'
with open(new_anno_path, 'w') as fd:
    json.dump(data_val, fd)
    

