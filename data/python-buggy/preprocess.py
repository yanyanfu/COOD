##
# concatenate code tokens directly


import json
import tqdm
import random
import numpy as np

DATASET_PATH = './test_python_buggy.jsonl'


dataset = []
with open(DATASET_PATH) as f:
    for line in f:
        line=line.strip()
        js=json.loads(line)
        dataset.append (js)


random.seed(1)
def format_str(string):
    for char in ['\r\n', '\r', '\n']:
        string = string.replace(char, ' ')
    return string
lang = 'java'


train_data = []
docstring_tokens_len = []
idx = 1
for js in dataset:
    docstring_tokens_len.append(len(js['docstring_tokens']))
    code = " ".join(js['code_tokens'])
    train_data.append({'idx': idx,
                       'doc': " ".join(js['docstring_tokens']),
                       'code': format_str(code),
                       'raw': js['original_string'],
                       'url': js['url'],
                       'label': 1})
    idx += 1

print(np.mean(docstring_tokens_len))
print(np.std(docstring_tokens_len))

to_train_file = './test_buggy_0.json'
with open(to_train_file, 'w', encoding='utf-8') as fp:
    json.dump(train_data, fp, indent=1)


