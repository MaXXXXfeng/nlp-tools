#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : find_new_words.py
# @Author  : Max
# @Time    : 2021/4/4
# @Desc: 新词发现

from smoothnlp.algorithm.phrase import extract_phrase


def find_new_words(corpus, num):
    '''
    extract_phrase 参数说明
    corpus:     必需，fileIO、database connection或list
                example:corpus = open(file_name, 'r', encoding='utf-8')
                        corpus = conn.execute(query)
                        corpus = list(***)
    top_k:      float or int,表示短语抽取的比例或个数
    chunk_size: int,用chunksize分块大小来读取文件
    min_n:      int,抽取ngram及以下
    max_n:      int,抽取ngram及以下
    min_freq:   int,抽取目标的最低词频
    '''
    new_words = extract_phrase(corpus=corpus, top_k=num)
    return new_words
