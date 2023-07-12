# -*- coding: utf-8 -*-
# data_utils
import json
import os
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, List, Dict, Any
import random
import copy
from dataclasses import dataclass
from transformers import AutoTokenizer
from itertools import combinations
from collections import defaultdict

LABEL2TEXT = {0:'entailment', 1:'neutral', 2:'contradiction'}


@dataclass
class FiDDataset(Dataset):
    data:List[dict]
    tokenizer:AutoTokenizer
    max_length:Optional[int]=None
    answer_max_length:Optional[int]=None
    n_docs:Optional[int]=1
    # premise : List[str] # hypothesis : str
    # input : 
    # label : 
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)
    
    def collate_fn(self, batch):
        inputs = []
        labels = []
        for b in batch:
            docs = b['premise'][:self.n_docs]
            for d in docs:
                inputs.append('premise: '+d+'hypothesis: '+b['hypothesis'])
            if b.get('label') is not None:
                labels.append(LABEL2TEXT[b['label']])            
        if self.max_length is None:
            inputs = self.tokenizer(inputs, padding='longest',return_tensors = 'pt')
        else:
            inputs = self.tokenizer(inputs, padding=True, truncation=True, max_length=self.max_length, return_tensors = 'pt')
        if labels:
            if self.max_length is None:
                labels = self.tokenizer(labels, padding='longest',return_tensors = 'pt').input_ids
            else:
                labels = self.tokenizer(labels, padding=True, truncation=True, max_length=self.answer_max_length, return_tensors = 'pt').input_ids
            inputs.data['labels']=labels
        
        inputs.data['input_ids']=inputs.data['input_ids'].reshape(len(batch), self.n_docs, -1)
        inputs.data['attention_mask']=inputs.data['attention_mask'].reshape(len(batch), self.n_docs, -1)
        return inputs