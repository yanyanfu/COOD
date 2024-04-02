#!/usr/bin/env python
# coding: utf-8

# In[52]:


import json
import random
import tqdm
import os
import copy
import sys
import numpy as np

random.seed(1)
def format_str(string):
    for char in ['\r\n', '\r', '\n']:
        string = string.replace(char, ' ')
    return string
lang = 'java'

train=[]
valid=[]
for root, dirs, files in os.walk(lang+'/final'):
    for file in files:
        temp=os.path.join(root,file)
        if '.jsonl' in temp:
            if 'train' in temp:
                train.append(temp)
            if 'valid' in temp:
                valid.append(temp)
            if 'test' in temp:
                valid.append(temp)


data={}                    
for file in train:
    if '.gz' in file:
        os.system("gzip -d {}".format(file))
        file=file.replace('.gz','')
    with open(file) as f:
        for line in f:
            line=line.strip()
            js=json.loads(line)
            data[js['url']]=js
print(len(data))

urls = []
with open("train.txt", 'r') as f1:
    for line in f1:
        line = line.strip()
        urls.append(line)
raw_data_train = {url:data[url] for url in urls}
print(len(raw_data_train))

train_data = []
docstring_tokens_len = []
code_tokens_len = []
idx = 1
for url, js in raw_data_train.items():
    docstring_tokens_len.append(len(js['docstring_tokens']))
    code_tokens_len.append(len(js['code_tokens']))
    code = " ".join(js['code_tokens'])
    train_data.append({'idx': idx,
                       'doc': " ".join(js['docstring_tokens']),
                       'code': format_str(code),
                       'raw': js['original_string'],
                       'url': url,
                       'label': 1})
    idx += 1

print(np.mean(docstring_tokens_len))
print(np.std(docstring_tokens_len))
print(np.mean(code_tokens_len))
print(np.std(code_tokens_len))

# num_negative = int(sys.argv[1])
num_negative = 0
train_data_withneg = copy.deepcopy(train_data)
print(len(train_data_withneg))
if num_negative > 0:
    for idx_x in tqdm.tqdm(range(length)):
        random_selected = random.sample(train_data[:idx_x]+train_data[idx_x+1:length], num_negative)
        for i in range(num_negative):
            train_data_withneg.append({'idx': idx_x+length+1,
                                    'doc': train_data[idx_x]['doc'],
                                    'code': random_selected[i]['code'],
                                    'label': 0})
print(len(train_data_withneg))

to_train_file = './train_codesearchnet_{}.json'.format(num_negative)
with open(to_train_file, 'w', encoding='utf-8') as fp:
    json.dump(train_data_withneg, fp, indent=1)


###################
# data={}
# for file in valid:
#     if '.gz' in file:
#         os.system("gzip -d {}".format(file))
#         file=file.replace('.gz','')
#     with open(file) as f:
#         for line in f:
#             line=line.strip()
#             js=json.loads(line)
#             data[js['url']]=js
# print(len(data))

# urls = []
# with open("test.txt", 'r') as f1:
#     for line in f1:
#         line = line.strip()
#         urls.append(line)
# raw_data_valid = {url:data[url] for url in urls if url in data}
# print(len(raw_data_valid))

# valid_data = []
# idx = 1
# for url, js in raw_data_valid.items():
#     code = " ".join(js['code_tokens'])
#     valid_data.append({'idx': idx,
#                        'doc': " ".join(js['docstring_tokens']),
#                        'code': format_str(code),
#                        'raw': js['original_string'],
#                        'url': url,
#                        'label': 1})
#     idx += 1
# print(len(valid_data))

# to_valid_file = './dev_codesearchnet.json'
# with open(to_valid_file, 'w', encoding='utf-8') as fp:
#     json.dump(valid_data, fp, indent=1)


# urls_codebase = []
# with open("codebase.txt", 'r') as f1:
#     for line in f1:
#         line = line.strip()
#         urls_codebase.append(line)
# raw_data_codebase = {url:data[url] for url in urls_codebase if url in data and url not in urls}
# print(len(raw_data_codebase))

# codebase_data = []
# idx = 1
# for url, js in raw_data_codebase.items():
#     code = " ".join(js['code_tokens'])
#     codebase_data.append({'idx': idx,
#                        'doc': " ".join(js['docstring_tokens']),
#                        'code': format_str(code),
#                        'raw': js['original_string'],
#                        'url': url,
#                        'label': 1})
#     idx += 1
# print(len(codebase_data))

# to_codebase_file = './dev_codebase.json'
# with open(to_codebase_file, 'w', encoding='utf-8') as fp:
#     json.dump(codebase_data, fp, indent=1)


## test scenario 2
uni_preds = json.load(open('unixcoder_preds.json', 'r'))
preds = []
for row in uni_preds:
    preds_dict = {}
    url = row['url']
    p_url = row['predicted_url']
    preds_dict['id'] = 'test-scenario2-{}'.format(row['idx'])
    preds_dict['text'] = " ".join(data[url]['docstring_tokens'])
    preds_dict['code'] = " ".join(data[p_url]['code_tokens'])
    preds_dict['aligned_code'] = " ".join(data[url]['code_tokens'])
    preds_dict['label'] = row['label']
    preds_dict['raw'] = ''
    preds.append(preds_dict)

np.random.shuffle(preds)
print(len(preds))
with open('data_test_scenario2.json', 'w') as f:
    json.dump(preds, f, indent=2)