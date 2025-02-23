import os 
import json 
import ipdb 

anno_path = '/data1/niehui/lvis/annotations/lvis_v1_val.json'
with open(anno_path) as fd:
    data = json.load(fd)
# print(data['categories'][0])
# {'image_count': 8, 'synonyms': ['aerosol_can', 'spray_can'], 'def': 'a dispenser that holds a substance under pressure', 'id': 1, 'synset': 'aerosol.n.02', 'name': 'aerosol_can', 'frequency': 'c', 'instance_count': 11}
classname_set = {'f':[], 'c':[], 'r':[], 'f_and_c':[], 'f_and_c_and_r':[]} # frequent, common, rare
classid_set = {'f':[], 'c':[], 'r':[], 'f_and_c':[], 'f_and_c_and_r':[]} # frequent, common, rare
for c in data['categories']:
    classname_set[c['frequency']].append(c['name'])
    classid_set[c['frequency']].append(c['id'])
    if c['frequency'] == 'f' or c['frequency'] == 'c':
        classname_set['f_and_c'].append(c['name'])
        classid_set['f_and_c'].append(c['id'])
    if c['frequency'] == 'f' or c['frequency'] == 'c' or c['frequency'] == 'r':
        classname_set['f_and_c_and_r'].append(c['name'])
        classid_set['f_and_c_and_r'].append(c['id'])
print(len(classid_set['f']))
print(len(classid_set['c']))
print(len(classid_set['r']))
ipdb.set_trace()
print(classname_set['f_and_c_and_r'])
print(classname_set['f_and_c'])
print(classname_set['r'])
