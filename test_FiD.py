# test FiD
# -*- coding: utf-8 -*-
import os
import json
import pickle
import copy
from tqdm import tqdm
import time
import re
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader,DistributedSampler,RandomSampler,SequentialSampler
import numpy as np
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
from tokenizers import AddedToken #XXX
from utils.utils_data import BYTE_TOKENS #XXX
import argparse
from utils.data_utils import *
from utils.distributed_utils import *
from utils.utils import *
from utils.metrics import *
from fid_model import *
import pandas as pd

def get_args():
    # parser
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument('--test_data', type=str, help = 'test_data 위치')
    parser.add_argument('--output_dir', type=str, help = 'output 위치')
    parser.add_argument('--name', type=str, help = 'name')
    
    # PTM model
    parser.add_argument('--ptm_path', type=str)
    parser.add_argument('--model_path', type=str)
    
    # model
    parser.add_argument('--n_docs', type=int, default = 1)
    parser.add_argument('--max_length', type=int)
    parser.add_argument('--answer_max_length', type=int)
    
    # TODO
    ## distributed 관련
    parser.add_argument('--local_rank', type=int, default = -1)
    parser.add_argument('--distributed', type=str2bool, default = False)
    args = parser.parse_args()
    return args

def evaluation(args, model, tokenizer, eval_dataloader):
    total_loss = 0.
    model.eval()
    Predict = []
    Actual = []
    with torch.no_grad():
        for data in tqdm(eval_dataloader, desc = 'evaluate', disable =  args.local_rank not in [-1,0]):
            labels = copy.deepcopy(data['labels'])
            data = {i:j.cuda() for i,j in data.items()}
            data['labels'][data['labels']==tokenizer.pad_token_id]=-100 # 굉장히 중요.
            output = model.forward(**data)
            loss = output['loss']
            total_loss+=loss
            data.pop('labels')
            model_to_generate = model.module if hasattr(model,'module') else model
            outputs = model_to_generate.generate(
                    **data,
                    max_length = args.answer_max_length,
                    pad_token_id = tokenizer.pad_token_id,
                    decoder_start_token_id=tokenizer.pad_token_id,
                    bos_token_id=tokenizer.eos_token_id,
                    eos_token_id = tokenizer.eos_token_id,
                    early_stopping = True,
                    do_sample = False,
                    num_beams = 20,
					)
            predicts = tokenizer.batch_decode(outputs, skip_special_tokens = True)
            actuals = tokenizer.batch_decode(labels, skip_special_tokens = True)
            Predict.extend(predicts)
            Actual.extend(actuals)
    # exact match
    acc = []
    for i,j in zip(Predict, Actual):
        acc.append(i==j)
    cnt = len(Predict)
    return dict(Loss=total_loss/len(eval_dataloader), cnt=cnt, acc=acc), Predict, Actual

if __name__=='__main__':
    args  = get_args()
    print(args)
    seed_everything(42)
    os.makedirs(args.output_dir, exist_ok = True)    
    # tokenizer, config, model
    ###########################################################################################
    tokenizer = T5Tokenizer.from_pretrained(args.ptm_path, extra_ids=0)
    config = T5Config.from_pretrained(args.ptm_path)
    model = FiDT5(config)
    model.load_state_dict(torch.load(args.model_path))
    
    if args.local_rank == -1:  # single-node multi-gpu (or cpu) mode
        device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
    # multi gpu
    else:
        device = torch.device(f'cuda:{args.local_rank}')
        torch.cuda.set_device(device) 
        model.to(device)
    ###########################################################################################################################################
    # data load
    test_data = load_data(args.test_data, args.local_rank, args.distributed)
    test_dataset = FiDDataset(args, test_data, tokenizer, args.is_train, args.shuffle)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset,batch_size = args.batch_size, sampler = test_sampler,collate_fn=test_dataset._collate_fn)
    ###########################################################################################################################################
    
    
    ###########################################################################################################################################
    # evaluation
    scores, predicts, actuals = evaluation(args, model, tokenizer, test_data, test_dataloader)
    print(scores)
    ###########################################################################################################################################
    
    ###########################################################################################################################################
    # save
    for i,j in zip(test_data, predicts):
        i['predict']=j
    
    save_jsonl(args.output_dir, test_data, args.name)
    
    with open(os.path.join(args.output_dir, args.name+'.txt')) as f:
        f.write(str(scores))
    ###########################################################################################################################################
