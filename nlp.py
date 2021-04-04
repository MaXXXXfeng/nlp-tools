#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : nlp.py
# @Author  : Max
# @Time    : 2021/4/4
# @Desc: 集成常用的nlp方法

from gensim.models import KeyedVectors
from wordcloud import WordCloud
import imageio
from transformers import BertModel, BertTokenizer

from config import CONF
from tasks.tokenize import cut
from tasks.train_vector import train_word2vec
from tasks.word2vec import get_w2v_vector, get_bert_vector
from tasks.find_new_words import find_new_words


class Doraemon:
    def __init__(self, pre_load_w2v=False, pre_load_bert=True, lang='cn'):
        self.cut = cut
        self.lang = lang

        # 加载bert信息
        if pre_load_bert:
            self.load_bert()
        else:
            self.bert = None
            self.tokenizer = None

        # 加载word2vec信息
        if pre_load_w2v:
            self.load_w2v()
        else:
            self.w2v_model = None

    def load_bert(self):
        '''加载bert相关信息'''
        print('加载bert模型')
        model = CONF.BERT_MODEL.get(self.lang,'bert-base-chinese')
        self.tokenizer = BertTokenizer.from_pretrained(model)
        self.bert = BertModel.from_pretrained(model)
        print('bert模型加载完成')

    def load_w2v(self,model_path=None):
        '''加载word2vec相关信息'''
        print('加载word2vec模型,请耐心等待')
        if not model_path:
            model_path = CONF.WORD2VEC_MODEL_PATH
        else:
            model_path = model_path
        if model_path.endswith('.txt'):
            self.w2v_model = KeyedVectors.load_word2vec_format(model_path,binary=False)
        else:
            self.w2v_model = KeyedVectors.load_word2vec_format(model_path)
        print('word2vec模型加载完成')

    def train_vector(self, input_path, model_path, size=100, use_cut=True, use_binary=True, mode='w2v'):
        '''
        根据自定义语料训练词向量
        :param input_path: 语料路径，txt格式
        :param model_path: 模型保存路径
        :param size: 词向量维度。默认100
        :param use_cut: 是否分词，不分词则模型默认以空格分隔。默认分词。
        :param use_binary: 是否保存为二进制模型文件，默认为true。Flase则保存为txt格式
        :param mode: 词向量训练模式，默认word2vec
        :return:
        '''
        accept_mode = ['w2v']
        if mode not in accept_mode:
            raise Exception(f"仅支持{accept_mode}词向量训练模式")
        if mode == 'glove':
            # TODO:添加Python版本的glove训练方法
            pass
        train_word2vec(input_path, model_path, size=size, use_cut=use_cut, use_binary=use_binary)

    def get_vector(self, inputs, mode='w2v'):
        '''
        获取词向量
        :param inputs:字符。单词或句子(bert模式下)
        :param mode: w2v:word2vec模式，bert:bert模式
        :return: 词向量。bert模式下会返回隐层向量和cls向量。请参考readme介绍。
        '''
        accept_mode = ['w2v', 'bert']
        if mode not in accept_mode:
            raise Exception("词向量模式不支持")
        if not isinstance(inputs, str):
            raise Exception('词向量模式仅支持字符串格式输入')
        if mode == 'bert':
            if not self.bert:
                self.load_bert()
            outputs = get_bert_vector(inputs=inputs, tokenizer=self.tokenizer, bert=self.bert)
            return outputs
        if not self.w2v_model:
            self.load_w2v()
        vector = get_w2v_vector(inputs, self.w2v_model)
        return vector

    def compute_similarity(self,word1,word2):
        '''计算两个词的相似度'''
        if not (isinstance(word1,str) and isinstance(word2,str)):
            raise Exception("输入格式错误")
        if not self.w2v_model:
            self.load_w2v()
        try:
            similarity = self.w2v_model.similarity(word1,word2)
        except KeyError as e:
            return -1
        return similarity

    def find_most_similar_words(self,word,K=10):
        '''计算与种子词相似度最高的K个词'''
        if (not isinstance(word,str)) and (not isinstance(word,list)):
            raise Exception("输入格式错误")
        if not self.w2v_model:
            self.load_w2v()
        try:
            similar_words = self.w2v_model.most_similar(positive=word,topn=K)
        except KeyError as e:
            similar_words = []
        return similar_words

    def find_new_words(self,inputs,words_num=200):
        '''
        对文本进行新词发现
        :param inputs: txt文本文件路径或[str,str...]
        :param words_num: 返回的新词数量,默认200
        :return: [str,str]
        '''
        if isinstance(inputs,str):
            if not inputs.endswith('.txt'):
                raise Exception("文本数据格式错误,仅支持txt格式")
            corpus = open(inputs, 'r', encoding='utf-8')
        elif isinstance(inputs,list):
            corpus = inputs
        else:
            raise Exception("输入格式错误")
        top_new_words = find_new_words(corpus=corpus,num=words_num)
        return top_new_words

    def create_word_cloud(self,inputs,output_path,bg_img=None,color='black'):
        '''
        基于语料生成词云
        :param inputs: 输入语料
        :param output_path: 图片保存路径
        :param bg_img: str,图片路径，修改背景图片,默认矩形
        :param color: str,背景颜色,默认黑色
        :return:
        '''
        if not isinstance(inputs,str):
            raise Exception("输入格式错误")
        if inputs.endswith('.txt'):
            corpus = open(inputs, 'r', encoding='utf-8')
            txt = corpus.read()
        else:
            txt = inputs
        if not output_path:
            output_path = 'data/word_cloud.png'
        if bg_img:
            bg_img = imageio.imread(bg_img)

        generator = WordCloud(font_path=CONF.FONT_PATH,mask=bg_img,background_color=color)
        generator.generate(txt)
        generator.to_file(output_path)
