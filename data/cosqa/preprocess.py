import json
import os
import numpy as np
import regex as re
import pandas as pd

np.random.seed(0)
def format_str(string):
    for char in ['\r\n', '\r', '\n']:
        string = string.replace(char, ' ')
    return string
lang = 'java'

train_raw = json.load(open('cosqa-retrieval-train-19604.json', 'r'))

train = []
for row in train_raw:
    train_dict = {}
    train_dict['id'] = 'train-python-{}'.format(row['idx'])
    train_dict['code'] = row['code_tokens']
    train_dict['text'] = row['doc']
    train_dict['label'] = row['label']
    train_dict['raw'] = row['code']
    if row['label'] == 1:
        train.append(train_dict)

test_raw = json.load(open('cosqa-retrieval-dev-500.json', 'r'))
test_raw += json.load(open('cosqa-retrieval-test-500.json', 'r'))

test = []
for row in test_raw:
    test_dict = {}
    test_dict['id'] = 'test-python-{}'.format(row['idx'])
    test_dict['code'] = row['code_tokens']
    test_dict['text'] = row['doc']
    test_dict['label'] = row['label']
    test_dict['raw'] = row['code']
    if row['label'] == 1:
        test.append(test_dict)

np.random.shuffle(train)
np.random.shuffle(test)

valid_size = int(len(train) * 0.1)
valid = train[:valid_size]
train = train[valid_size:]

with open('data_train.json', 'w') as f:
    json.dump(train, f, indent=2)

with open('data_valid.json', 'w') as f:
    json.dump(valid, f, indent=2)

with open('data_test.json', 'w') as f:
    json.dump(test, f, indent=2)

print(len(train))
print(len(valid))
print(len(test))

print(json.dumps(train[:2], indent=4))
print(json.dumps(valid[:2], indent=4))
print(json.dumps(test[:2], indent=4))