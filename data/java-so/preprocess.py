import sys
import json
import re
import tqdm
import random

import numpy as np

from parsers import (remove_comments_and_docstrings,
                   tree_to_token_index,
                   index_to_code_token,
                   tree_to_variable_index)
from tree_sitter import Language, Parser
from typing import List
from collections import defaultdict


# lang = 'java'
# LANGUAGE = Language('../../data/parsers/my-languages.so', lang)
# parser = Parser()
# parser.set_language(LANGUAGE) 


# def extract_tokens(code, parser):
#     #remove comments
#     try:
#         code=remove_comments_and_docstrings(code,lang)
#         tree = parser.parse(bytes(code,'utf8'))    
#         root_node = tree.root_node  
#         tokens_index=tree_to_token_index(root_node)     
#         code=code.split('\n')
#         code_tokens=[index_to_code_token(x,code) for x in tokens_index]
#         code_tokens_2=[x for x in code_tokens if x]
#         return code_tokens_2
#     except:
#         pass
    


# def node_to_string(lines, node) -> str:
#     start_point = node.start_point
#     end_point = node.end_point
#     if start_point[0] == end_point[0]:
#         return lines[start_point[0]][start_point[1]:end_point[1]]
#     ret = lines[start_point[0]][start_point[1]:] + "\n"
#     ret += "\n".join([line for line in lines[start_point[0] + 1:end_point[0]]])
#     ret += "\n" + lines[end_point[0]][:end_point[1]]
#     return ret


# def tokenize_docstring(docstring: str) -> List[str]:
#     DOCSTRING_REGEX_TOKENIZER = re.compile(r"[^\s,'\"`.():\[\]=*;>{\}+-/\\]+|\\+|\.+|\(\)|{\}|\[\]|\(+|\)+|:+|\[+|\]+|{+|\}+|=+|\*+|;+|>+|\++|-+|/+")
#     return [t for t in DOCSTRING_REGEX_TOKENIZER.findall(docstring) if t is not None and len(t) > 0]


# def compare_length(s):
#     return len(s)


# language_library = Language('./my-languages.so', lang)
# PARSER_m = Parser()
# PARSER_m.set_language(language_library)
# query = language_library.query("""
#         (method_declaration) @method
#         """)

# to_train_file = './train_so.json'
# file_path = './sodata-ja.jsonl'
# data=[]

# with open(to_train_file, 'w', encoding='utf-8') as fp:
#     with open(file_path) as f:
#         for line in f:
#             line=line.strip()
#             js=json.loads(line)
#             tree = PARSER_m.parse(bytes(js['a_code'], "utf8"))
#             lines = js['a_code'].split('\n')
#             captures = query.captures(tree.root_node)
#             cur_method_nodes = [node[0] for node in captures if node[1] == 'method']
#             js['a_code'] = [node_to_string(lines, node) for node in cur_method_nodes]
#             if js['a_code']:
#                 code_tokens = []
#                 for func in js['a_code']:
#                     tokens = extract_tokens(func, parser)
#                     if tokens:
#                         code_tokens.append(tokens)            
#                 if code_tokens:
#                     s = {}
#                     code_tokens = sorted(code_tokens, key=compare_length)
#                     s['code_tokens'] = code_tokens[-1]
#                     s['docstring_tokens'] = tokenize_docstring(js['q_title'])

#                     data.append(s)
#                     fp.write(json.dumps(s)+'\n')
# print(len(data))

## statistics on mean and dev. of original set
## sampling with the same mean and dev.
## token length statistics
## double check length calculation


# DATASET_PATH = 'train_so.json'

# dataset = []
# with open(DATASET_PATH) as f:
#     for line in f:
#         line=line.strip()
#         js=json.loads(line)
#         dataset.append (js)

# random.seed(1)
# def format_str(string):
#     for char in ['\r\n', '\r', '\n']:
#         string = string.replace(char, ' ')
#     return string
# lang = 'java'


# dt = defaultdict(list)
# for data in dataset:
#     dt[len(data['docstring_tokens'])].append(data)

# sampling = {}
# for i in range(8000):
#     s = round(random.gauss(13.251778, 10.028015))
#     if s not in sampling.keys():
#         sampling[s] = 1
#     else:
#         sampling[s] += 1

# sampled_dataset =[]
# for key, value in sampling.items():
#     if key in dt.keys():
#         m = min(value, len(dt[key]))
#         sampled_dataset.extend(dt[key][:m])
# print(len(sampled_dataset))


# train_data = []
# docstring_tokens_len = []
# code_tokens_len = []
# idx = 1
# for js in sampled_dataset:
#     docstring_tokens_len.append(len(js['docstring_tokens']))
#     code_tokens_len.append(len(js['code_tokens']))
#     code = " ".join(js['code_tokens'])
#     train_data.append({'idx': idx,
#                        'doc': " ".join(js['docstring_tokens']),
#                        'code': format_str(code),
#                        'raw': '',
#                        'url': '',
#                        'label': 1})
#     idx += 1
# print(np.mean(docstring_tokens_len))
# print(np.std(docstring_tokens_len))
# print(np.mean(code_tokens_len))
# print(np.std(code_tokens_len))

# to_train_file = './train_so_0.json'
# with open(to_train_file, 'w', encoding='utf-8') as fp:
#     json.dump(train_data, fp, indent=1)

DATASET_PATH = 'train_so.json'
dataset = []
with open(DATASET_PATH) as f:
    for line in f:
        line=line.strip()
        js=json.loads(line)
        dataset.append (js)


# DATASET_PATH = 'train_so_0.json'
# dataset0 = []
# with open(DATASET_PATH) as f:
#     for line in f:
#         line=line.strip()
#         js=json.loads(line)
#         dataset0.append (js)

random.seed(1)
def format_str(string):
    for char in ['\r\n', '\r', '\n']:
        string = string.replace(char, ' ')
    return string
lang = 'java'

idx = 1
train_data = []
# for js in dataset:
#     code = " ".join(js['code_tokens'])
#     code = format_str(code)
#     flag = 1
#     for js0 in dataset0:    
#         if code == js0['code'] :
#             flag = 0
#             break
#     if flag:
#         train_data.append({'idx': idx,
#                         'doc': " ".join(js['docstring_tokens']),
#                         'code': code,
#                         'raw': '',
#                         'url': '',
#                         'label': 1})
#         idx += 1
#         if idx == 4965:
#             break

for js in dataset:
    code = " ".join(js['code_tokens'])
    train_data.append({'idx': idx,
                       'doc': " ".join(js['docstring_tokens']),
                       'code': format_str(code),
                       'raw': '',
                       'url': '',
                       'label': 1})
    idx += 1

to_train_file = './train_so_1.json'
with open(to_train_file, 'w', encoding='utf-8') as fp:
    json.dump(train_data, fp, indent=1)

