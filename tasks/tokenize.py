#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : tokenize.py
# @Author  : Max
# @Time    : 2021/4/4
# @Desc: 中文分词工具
import jieba

from uils import check_file_suffix

def cut_one_sentence(sentence,delimiter=' '):
    '''对一句话进行分词'''
    seg = jieba.cut(sentence)
    seg = delimiter.join(list(seg))
    return seg

def cut_txt(f_path,out_path=None):
    '''对文本文件进行分词'''
    if not out_path:
        out_path = f_path.replace('.txt', '_cut.txt')
    with open(f_path,'rb') as f:
        document = f.read()
        document_cut = jieba.cut(document)
        result = ' '.join(document_cut)
        result = result.encode('utf-8')
        with open(out_path,'wb+') as f2:
            f2.write(result)
    f.close()
    f2.close()
    print(f'文件分词完成,路径{out_path}')
    return out_path

def cut(sentence,delimiter=' ',is_file=False,output_path=None):
    '''
    分词
    :param sentence: 待分词内容。支持单句(str)或多句(list(str))
    :param delimiter: 分隔符，默认使用空格分隔
    :param is_file: 是否为文件分词。若为true,sentence为文件路径
    :param output_path: 文件输出路径，默认输出在原文件路径。
    :return:
    '''
    if is_file:
        if not check_file_suffix(sentence):
            raise Exception('当前仅支持txt文件，请检查文件路径')
        result = cut_txt(sentence, output_path)
        return result
    if isinstance(sentence,str):
        return cut_one_sentence(sentence,delimiter)
    elif isinstance(sentence,list):
        return [[cut_one_sentence(sen)] for sen in sentence]
    raise Exception('不支持的输入格式')

if __name__=='__main':
    pass
