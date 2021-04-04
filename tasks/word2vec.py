#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : word2vec.py
# @Author  : Max
# @Time    : 2021/4/4
# @Desc: 词向量相关
import torch
from transformers import BertModel, BertTokenizer


def get_w2v_vector(word, model):
    if word in model:
        return model[word]
    return -1


def get_bert_vector(inputs, tokenizer, bert):
    inputs = tokenizer(inputs, return_tensors="pt")
    output = bert(**inputs)
    last_hidden_state = output[0]  # batch_size * seq_length * hidden_size
    pooler_output = output[1]  # batch_size * hidden_size, first token(cls)
    np_sen = last_hidden_state.detach().numpy()
    np_cls = pooler_output.detach().numpy()
    return (np_sen, np_cls)
