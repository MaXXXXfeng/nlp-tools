#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : train_vector.py
# @Author  : Max
# @Time    : 2021/4/4
# @Desc:   训练词向量
from gensim.models import word2vec

from uils import check_file_suffix
from tasks.tokenize import cut_txt


def train_word2vec(f_path, model_path, size=100,use_cut=True, use_binary=True):
    '''
    使用Word2vec训练自定义词向量
    参数说明：
    f_path:语料路径，txt格式
    model_path:模型保存路径
    use_cut:默认分词，不分词则模型默认以空格分隔
    size:词向量维度。默认100
    use_binary:默认保存为二进制模型文件

    Word2Vec对应参数说明：
    sentence：待分析的语料
    size:词向量维度。默认100
    window：词向量上下文最大距离。默认5，推荐值[5,10]
    sg: 0-CBOW，1-Skip-Gram, 默认CBOW模型
    hs: 0-Negative Sampling, 1-Hierachical Softmax, 默认Negative Sampling
    min_count: 需要计算词向量的最小词频。 默认5
    iter:迭代次数。默认5。大语料建议增加。

    '''
    if not check_file_suffix(f_path,['.txt']):
        raise Exception('语料文件格式仅支持txt')
    if use_cut:
        f_path = cut_txt(f_path)
    sentences = word2vec.LineSentence(f_path)
    model = word2vec.Word2Vec(sentences, hs=1, min_count=1, window=3, vector_size=size)
    if use_binary:
        model.save(model_path)  # 模型保存为二进制文件
    else:
        if not check_file_suffix(model_path,accept_suffix=['txt']):
            raise Exception('模型路径需要为txt格式')
        model.wv.save_word2vec_format(model_path, binary=False)  # 模型保存为txt文件
    print('词向量训练完成')
