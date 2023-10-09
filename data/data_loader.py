import os
import copy
import subprocess
import numpy as np
import pandas as pd
import json
from typing import Any, Union, List
from tqdm import tqdm
from collections import Counter
# import code search net dataset from huggingface
# from datasets import load_dataset

import torch
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset, random_split
from torch.utils.data.distributed import DistributedSampler

from transformers import RobertaTokenizer
import spacy
nlp = spacy.load('en_core_web_sm')
# import torchtext

from data.parser import DFG_python,DFG_java,DFG_ruby,DFG_go,DFG_php,DFG_javascript
from data.parser import (remove_comments_and_docstrings,
                   tree_to_token_index,
                   index_to_code_token,
                   tree_to_variable_index)
from data.stopwords import STOPWORDS_PYTHON, STOPWORDS_JAVA
# from baseline.bpe import BpeVocabulary
from spacy.lang.en import English
from spacy.tokens.token import Token

from tree_sitter import Language, Parser
dfg_function={
    'python':DFG_python,
    'java':DFG_java,
    'ruby':DFG_ruby,
    'go':DFG_go,
    'php':DFG_php,
    'javascript':DFG_javascript
}

#load parsers
PARSER = {}        
for lang in dfg_function:
    LANGUAGE = Language('data/parser/my-languages.so', lang)
    parser = Parser()
    parser.set_language(LANGUAGE) 
    parser = [parser,dfg_function[lang]]    
    PARSER[lang]= parser

def extract_dataflow(code, parser, lang):
    #remove comments
    try:
        code=remove_comments_and_docstrings(code,lang)
    except:
        pass    
    #obtain dataflow
    if lang=="php":
        code="<?php"+code+"?>"    
    try:
        tree = parser[0].parse(bytes(code,'utf8'))    
        root_node = tree.root_node  
        tokens_index=tree_to_token_index(root_node)     
        code=code.split('\n')
        code_tokens=[index_to_code_token(x,code) for x in tokens_index]  
        index_to_code={}
        for idx,(index,code) in enumerate(zip(tokens_index,code_tokens)):
            index_to_code[index]=(idx,code)  
        try:
            DFG,_=parser[1](root_node,index_to_code,{}) 
        except:
            DFG=[]
        DFG=sorted(DFG,key=lambda x:x[1])
        indexs=set()
        for d in DFG:
            if len(d[-1])!=0:
                indexs.add(d[1])
            for x in d[-1]:
                indexs.add(x)
        new_DFG=[]
        for d in DFG:
            if d[1] in indexs:
                new_DFG.append(d)
        dfg=new_DFG
    except:
        dfg=[]
    return code_tokens,dfg

def fixed_token_shuffle(idx, 
                        mask, 
                        percent=1.0):
    
    to_permute = [i for i, m in enumerate(mask) if m]
    num_permute = int(len(to_permute) * percent)
    num_to_fix = len(to_permute) - num_permute
    if num_to_fix > 0:
        to_fix_ids = np.random.choice(to_permute, num_to_fix, replace=False)
        mask = [True if mask[i] or i in to_fix_ids else False for i in range(len(mask))]

    unfrozen_indices = [i for i, e in enumerate(idx) if not mask[i]]
    unfrozen_set = idx[unfrozen_indices]
    unfrozen_set_og = copy.deepcopy(unfrozen_set)
    if len(unfrozen_set) > 1:
        while True:
            np.random.shuffle(unfrozen_set)
            same_pos = sum(
                [
                    1
                    for i, e in enumerate(unfrozen_set)
                    if unfrozen_set[i] == unfrozen_set_og[i]
                ]
            )
            if same_pos == 0:
                break
    idx[unfrozen_indices] = unfrozen_set

def randomize_shuffle(text,
                      retain_stop=False,
                      retain_punct=False,
                      percent=1.0,
                      lang="en",
                      tokenizer=None):

    tokens = tokenizer(text)
    masks = []
    masks.append(np.zeros(len(tokens)).astype(bool))
    if retain_punct:
        masks.append(
            np.array(
                [
                    t.is_punct if type(t) == Token else False
                    for t in tokens
                ]
            )
        )
    if retain_stop:
        masks.append(
            np.array(
                [
                    t.is_stop if type(t) == Token else False
                    for t in tokens
                ]
            )
        )
    mask = np.any(np.stack(masks), axis=0)
    indx = np.arange(len(tokens))
    fixed_token_shuffle(indx, mask, percent)
    new_tokens = [tokens[i] for i in indx]
    if lang == "en":
        separator = " "
    else:
        separator = ""
    new_text = separator.join([str(t) for t in new_tokens])
    if lang == "en":
        # removing extra space
        new_text = separator.join(new_text.split())
    # not possible for chinese as the tokenization words matter on randomization
    # de_rand = derandomize(new_text, indx, tokenizer)
    # assert de_rand == sent
    # percent changed
    ch = sum([1 for i, p in enumerate(indx) if i != p and not mask[i]]) / sum(
        [1 for m in mask if not m]
    )
    return new_text, indx, ch, mask

class AnomalyDataset(Dataset):
    def __init__(self, 
                 name, 
                 root,
                 mode = 'unsup',
                 code_col='code',
                 text_col='text',
                 label_col='label',
                 raw_col='raw',
                 tokenizer=None,
                 code_len=256,
                 text_len=128,
                 shuffle_ratio=None,
                 buggy_ratio=None,
                 wrong_text_ratio=None,
                 out_domain_ratio=None,
                 dataset_load = 'train',
                 baseline=False):

        self.name = name
        self.root = root
        self.mode = mode
        self.dataset_load = dataset_load
        self.baseline = baseline
        self.shuffle_ratio = shuffle_ratio
        self.buggy_ratio = buggy_ratio
        self.wrong_text_ratio = wrong_text_ratio
        self.out_domain_ratio = out_domain_ratio

        if self.dataset_load == 'train':
            self.data = pd.read_json(os.path.join(root, name, 'data_train.json'))
        elif self.dataset_load == 'valid':
            self.data = pd.read_json(os.path.join(root, name, 'data_valid.json'))
        elif dataset_load == 'test':
            self.data = pd.read_json(os.path.join(root, name, 'data_test.json'))
        elif dataset_load == 'codebase':
            self.data = pd.read_json(os.path.join(root, name, 'data_codebase.json'))
        else:
            self.data = pd.read_json(os.path.join(root, name, 'data_test_scenario2.json'))
            
        self.code_col = self.data.columns.get_loc(code_col)
        self.text_col = self.data.columns.get_loc(text_col)
        self.label_col = self.data.columns.get_loc(label_col)
        self.raw_col = self.data.columns.get_loc(raw_col)

        self.tokenizer = tokenizer
        self.code_len = code_len
        self.text_len = text_len
        self.anom_label = np.array([0]*len(self.data))

    def __getitem__(self, index):
        code = self.data.iloc[index, self.code_col]
        text = self.data.iloc[index, self.text_col]
        label = self.data.iloc[index, self.label_col]

        if self.anom_label[index] > 0:
            label = -1 * self.anom_label[index]

        if self.baseline:
            code_tokens = code.split()[:self.code_len]
            code_ids = self.tokenizer[0].numericalize([code_tokens]).squeeze(-1).tolist()
            code_length = len(code_ids)
            padding_length = self.code_len - code_length
            code_ids += [self.tokenizer[0].vocab.stoi['<pad>']] * padding_length
            code = torch.tensor(code_ids)

            text_tokens = text.split()[:self.text_len]
            text_ids = self.tokenizer[1].numericalize([text_tokens]).squeeze(-1).tolist()
            text_length = len(text_ids)
            padding_length = self.text_len - text_length
            text_ids += [self.tokenizer[1].vocab.stoi['<pad>']] * padding_length
            text = torch.tensor(text_ids)

            return code, text, code_length, text_length, label

        else:
            code_tokens = self.tokenizer.tokenize(code)[:self.code_len-4]
            code_tokens =[self.tokenizer.cls_token,"<encoder-only>",self.tokenizer.sep_token]+code_tokens+[self.tokenizer.sep_token]
            code_ids = self.tokenizer.convert_tokens_to_ids(code_tokens)
            padding_length = self.code_len - len(code_ids)
            code_ids += [self.tokenizer.pad_token_id] * padding_length
            code = torch.tensor(code_ids)

            text_tokens = self.tokenizer.tokenize(text)[:self.text_len-4]
            text_tokens = [self.tokenizer.cls_token,"<encoder-only>",self.tokenizer.sep_token]+text_tokens+[self.tokenizer.sep_token]
            text_ids = self.tokenizer.convert_tokens_to_ids(text_tokens)
            padding_length = self.text_len - len(text_ids)
            text_ids += [self.tokenizer.pad_token_id] * padding_length
            text = torch.tensor(text_ids)

            return code, text, label

    def __len__(self):
        return len(self.data)

class AnomalyDatasetWrapper(object):
    def __init__(self,
                 name,
                 root,
                 mode,
                 batch_size=32,
                 num_workers=4,
                 code_col='code',
                 text_col='text',
                 label_col='label',
                 raw_col='raw',
                 tokenizer=None,
                 code_len=256,
                 text_len=128,
                 anom_ratio=0.0,
                 shuffle_ratio=None,
                 buggy_ratio=None,
                 wrong_text_ratio=None,
                 out_domain_ratio=None,
                 anom_test=0.2,
                 baseline=False):

        self.name = name
        self.root = root
        self.mode = mode
        self.code_len = code_len
        self.text_len = text_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.code_col = code_col
        self.text_col = text_col
        self.label_col = label_col
        self.raw_col = raw_col

        if self.mode == 'unsup':
            self.shuffle_ratio = self.buggy_ratio = self.out_domain_ratio = self.wrong_text_ratio = 0.0
        else:
            self.shuffle_ratio = shuffle_ratio if shuffle_ratio is not None else anom_ratio / 4.0
            self.buggy_ratio = buggy_ratio if buggy_ratio is not None else anom_ratio / 4.0
            self.wrong_text_ratio = wrong_text_ratio if wrong_text_ratio is not None else anom_ratio / 4.0
            self.out_domain_ratio = out_domain_ratio if out_domain_ratio is not None else anom_ratio / 4.0

        self.anom_test = anom_test
        self.baseline = baseline
        
        if baseline:
            print('Building vocab counter on train set...')
            # def code_max_length(tokens):
            #     return tokens[:code_len]
            # def text_max_length(tokens):
            #     return tokens[:text_len]
            # code_tokenizer = torchtext.data.Field()
            # text_tokenizer = torchtext.data.Field()
            # data_train = torchtext.data.TabularDataset(path=os.path.join(root, name, 'data_vocab.json'),
            #                                            format='json', 
            #                                            fields={code_col: ('code', code_tokenizer), text_col: ('text', text_tokenizer)})
            # code_tokenizer.build_vocab(data_train, max_size=10000, min_freq=10)
            # text_tokenizer.build_vocab(data_train, max_size=10000, min_freq=10)
            # print(f'Code vocab size: {len(code_tokenizer.vocab):,}')
            # print(f'Text vocab size: {len(text_tokenizer.vocab):,}')
            # self.tokenizer = (code_tokenizer, text_tokenizer)
        else:
            self.tokenizer = RobertaTokenizer.from_pretrained(tokenizer if tokenizer is not None else 'roberta-base', do_lower_case=True)

    def _get_shuffled_sample(self, dataset):
        print('Shuffling codes...')

        shuffle_sample_size = max(int(len(dataset) * dataset.shuffle_ratio), 2)
        if shuffle_sample_size % 2 == 0:
            shuffle_sample_size -= 1
        shuffle_indices = np.random.choice(range(len(dataset)), shuffle_sample_size, replace=False)
        new_indices = np.flip(shuffle_indices)
        dataset.data.iloc[shuffle_indices, dataset.text_col] = dataset.data.iloc[new_indices, dataset.text_col].tolist()
        dataset.anom_label[shuffle_indices] = 2
        
        return dataset

    def _get_wrong_text_sample(self, dataset, lang='en'):
        print('Generating wrong queries...')
        non_anom_indices = [i for i in range(len(dataset)) if dataset.anom_label[i] == 0]
        wrong_text_sample_size = int(len(dataset) * dataset.wrong_text_ratio)
        wrong_text_indices = np.random.choice(non_anom_indices, wrong_text_sample_size, replace=False)

        if lang == "en":
            tokenizer = English().tokenizer
        else:
            raise NotImplementedError(f"Tokenizer for Language={lang} not implement")

        for i in tqdm(wrong_text_indices):
            try:
                text = dataset.data.iloc[i, dataset.text_col]
                wrong_text, _, change, _ = randomize_shuffle(text, lang=lang, retain_stop=True, retain_punct=True, percent=0.2, tokenizer=tokenizer)
                assert change > 0.0
                dataset.data.iloc[i, dataset.text_col] = wrong_text
                dataset.anom_label[i] = 3
            except:
                continue
        return dataset

    def _get_buggy_sample(self, dataset, buggy_dataset=None):
        if buggy_dataset is None:
            buggy_dataset = dataset.name + '-ood'
        print('Getting samples from {}...'.format(buggy_dataset))
        buggy_sample_size = int(len(dataset) * dataset.out_domain_ratio)
        buggy_sample = AnomalyDataset(buggy_dataset,
                                    self.root,
                                    mode=self.mode,
                                    code_col=self.code_col,
                                    text_col=self.text_col,
                                    raw_col=self.raw_col,
                                    label_col=self.label_col,
                                    tokenizer=self.tokenizer,
                                    code_len=self.code_len,
                                    text_len=self.text_len,
                                    dataset_load=dataset.dataset_load,
                                    baseline=dataset.baseline)
        
        if len(buggy_sample) > buggy_sample_size:
            buggy_indices = np.random.choice(len(buggy_sample), buggy_sample_size, replace=False)
            buggy_sample.data = buggy_sample.data.iloc[buggy_indices, :]

        non_anom_indices = [i for i in range(len(dataset)) if dataset.anom_label[i] == 0]
        replace_indices = np.random.choice(non_anom_indices, buggy_sample_size, replace=False)
        for i, j in tqdm(zip(replace_indices, range(len(buggy_sample))), total=len(buggy_sample)):
            dataset.data.iloc[i, dataset.code_col] = buggy_sample.data.iloc[j, buggy_sample.code_col]
            dataset.data.iloc[i, dataset.text_col] = buggy_sample.data.iloc[j, buggy_sample.text_col]
            dataset.anom_label[i] = 4
        return dataset

    def _get_out_domain_sample(self, dataset, anom_dataset):
        print('Getting samples from {}...'.format(anom_dataset))
        anom_sample_size = int(len(dataset) * dataset.out_domain_ratio)
        anom_sample = AnomalyDataset(anom_dataset,
                                self.root,
                                mode=self.mode,
                                code_col=self.code_col,
                                text_col=self.text_col,
                                raw_col=self.raw_col,
                                label_col=self.label_col,
                                tokenizer=self.tokenizer,
                                code_len=self.code_len,
                                text_len=self.text_len,
                                dataset_load=dataset.dataset_load,
                                baseline=dataset.baseline)
        
        if len(anom_sample) > anom_sample_size:
            anom_indices = np.random.choice(len(anom_sample), anom_sample_size, replace=False)
            anom_sample.data = anom_sample.data.iloc[anom_indices, :]
        # anom_sample.anom_label = [1] * len(anom_sample)

        non_anom_indices = [i for i in range(len(dataset)) if dataset.anom_label[i] == 0]
        replace_indices = np.random.choice(non_anom_indices, anom_sample_size, replace=False)
        for i, j in tqdm(zip(replace_indices, range(len(anom_sample))), total=len(anom_sample)):
            dataset.data.iloc[i, dataset.code_col] = anom_sample.data.iloc[j, anom_sample.code_col]
            dataset.data.iloc[i, dataset.text_col] = anom_sample.data.iloc[j, anom_sample.text_col]
            dataset.anom_label[i] = 1

        # new_dataset = Subset(dataset, np.setdiff1d(range(len(dataset)), replace_indices))
        # dataset = ConcatDataset([new_dataset, anom_sample])
        return dataset
    
    def get_train_loaders(self, 
                          anom_dataset='cosqa',
                          world_size=None,
                          rank=None,
                          dataset_load = 'train'):

        if world_size is not None and rank is not None:
            out_dir = 'buggy{}'.format(rank)
        else:
            out_dir = 'buggy0'

        dataset = AnomalyDataset(self.name,
                             self.root,
                             mode=self.mode,
                             code_col=self.code_col,
                             text_col=self.text_col,
                             label_col=self.label_col,
                             tokenizer=self.tokenizer,
                             code_len=self.code_len,
                             text_len=self.text_len,
                             shuffle_ratio=self.shuffle_ratio,
                             buggy_ratio=self.buggy_ratio,
                             wrong_text_ratio=self.wrong_text_ratio,
                             out_domain_ratio=self.out_domain_ratio,
                             dataset_load = dataset_load,
                             baseline=self.baseline)

        if dataset.shuffle_ratio > 0:
            dataset = self._get_shuffled_sample(dataset)

        if dataset.buggy_ratio > 0:
            dataset = self._get_buggy_sample(dataset)

        if dataset.wrong_text_ratio > 0:
            dataset = self._get_wrong_text_sample(dataset)

        if dataset.out_domain_ratio > 0:
            dataset = self._get_out_domain_sample(dataset, anom_dataset)

        print('Scenario 1: {}, Scenario 2: {}, Scenario 3: {}, Scenario 4: {}, ID: {}'.format(
            sum(dataset.anom_label == 1),
            sum(dataset.anom_label == 2),
            sum(dataset.anom_label == 3),
            sum(dataset.anom_label == 4),
            sum(dataset.anom_label == 0)
        ))

        if world_size is not None and rank is not None:
            train_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
            train_loader = DataLoader(dataset, batch_size=self.batch_size, pin_memory=True, sampler=train_sampler)
        
        else:
            train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, persistent_workers=True)

        return train_loader

    def get_test_loaders(self, 
                         test_dataset,
                         anom_dataset='cosqa',
                         test_type='anom',
                         dataset_load = 'test',
                         world_size=None,
                         rank=None):
        
        test_anom_ratio = self.anom_test
        if dataset_load != 'test':
            test_anom_ratio = 0.0

        test_shuffle_ratio = test_anom_ratio
        if test_type != 'anom':
            test_shuffle_ratio = 0.0

        dataset = AnomalyDataset(test_dataset,
                                self.root,
                                mode=self.mode,
                                code_col=self.code_col,
                                text_col=self.text_col,
                                label_col=self.label_col,
                                tokenizer=self.tokenizer,
                                code_len=self.code_len,
                                text_len=self.text_len,
                                shuffle_ratio=test_shuffle_ratio,
                                buggy_ratio=test_anom_ratio,
                                wrong_text_ratio=test_anom_ratio,
                                out_domain_ratio=test_anom_ratio,
                                dataset_load=dataset_load,
                                baseline=self.baseline)
        
        if dataset.shuffle_ratio > 0:
            dataset = self._get_shuffled_sample(dataset)
        if dataset.buggy_ratio > 0:
            dataset = self._get_buggy_sample(dataset)
        if dataset.wrong_text_ratio > 0:
            dataset = self._get_wrong_text_sample(dataset)
        if dataset.out_domain_ratio > 0:
            dataset = self._get_out_domain_sample(dataset, anom_dataset)

        print('Scenario 1: {}, Scenario 2: {}, Scenario 3: {}, Scenario 4: {}, ID: {}'.format(
            sum(dataset.anom_label == 1),
            sum(dataset.anom_label == 2),
            sum(dataset.anom_label == 3),
            sum(dataset.anom_label == 4),
            sum(dataset.anom_label == 0)
        ))

        if self.anom_test > 1.0:
            test = Subset(dataset, np.random.choice(len(dataset), int(self.anom_test), replace=False))
        else:
            test = dataset

        if world_size is not None and rank is not None:
            test_sampler = DistributedSampler(test, num_replicas=world_size, rank=rank, shuffle=False)
            test_loader = DataLoader(test, batch_size=self.batch_size, pin_memory=True, sampler=test_sampler)
        else:
            test_loader = DataLoader(test, batch_size=self.batch_size, shuffle=False)

        return test_loader