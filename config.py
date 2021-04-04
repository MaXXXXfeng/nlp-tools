#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : config.py
# @Author  : Max
# @Time    : 2021/4/4
# @Desc:

class Config(object):
    WORD2VEC_MODEL_PATH = ''
    BERT_MODEL = {
        'cn': 'bert-base-chinese',
        'en': 'bert-base-uncased'
    }
    FONT_PATH = 'data/simsun.ttf'



CONF = Config
