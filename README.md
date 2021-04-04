# nlp-tools
常见的nlp工具

## 1 安装

下载项目

```shell
git clone https://github.com/MaXXXXfeng/nlp-tools.git
cd nlp-tools
```

安装所需依赖

```shell
pip install -r requirements.txt
```

## 2 使用

各功能具体用法如下，使用前需要先初始化demo

```python
demo = Doraemon()
```

参数说明：

- pre_load_w2v: 是否预加载word2vec向量。默认不加载，如果需要用到词向量或相似度计算等可以提前加载。
- pre_load_bert: 是否预加载bert模型，默认加载。
- lang: 语言，默认cn。用于bert模型加载。

### 2.1 分词

对中文文本进行分词

- 方法名：cut()

- 参数

  - **sentence**: 待分词的文本，可以是字符或者字符列表
  - **delimiter**: 分词分隔符，默认为空格
  - **is_file**: 是否进行文件分词。默认为False。如果对文件分词，sentence需要为txt文件对应路径。
  - **output_path**: 文件分词结果输出路径。默认输出在原文件路径下。

- 返回值

  基于分隔符分隔的字符串

- 示例

  ```python
  test_sentence = '我来到清华大学'
  seg_words = demo.cut(test_sentence)
  ```
  
- 参考文献

  - [结巴中文分词](https://github.com/fxsjy/jieba)


### 2.2 训练词向量

通过自定义语料，训练词向量

- 方法名：train_vector()

- 参数

  - **input_path**: 语料路径，txt格式文件。
  - **model_path**: 模型文件输出路径。
  - **size**: 词向量维度。默认100。
  - **use_cut**: 是否对语料文件进行分词，默认进行分词。不分词则默认以语料中的空格为分隔符。
  - **use_binary**: 是否保存为二进制模型文件。默认为true。Fasle则保存为txt文件。
  - **mode**: 词向量训练模式，默认为w2v。暂不支持glove。

- 示例

  ```python
  input_file = '~/raw_corpus.txt'
  model1_path = '~/w2ve'
  model1_path = '~/w2ve.txt'
  demo.train_vector(input_path=input_file, model_path=model1_path) # 保存二进制模型文件
  demo.train_vector(input_path=input_file, model_path=model2_path,use_binary=False) # 保存txt模型文件
  
  ```

- 参考文献

  - [word2vec](https://en.wikipedia.org/wiki/Word2vec)
  - [使用gensim训练词向量](https://radimrehurek.com/gensim/models/word2vec.html)

### 2.3 词向量

获取指定词/句子的词向量

- 方法名：get_vector()

- 参数

  - **inputs**:字符串，中文单词。bert模式可对句子编码。
  - **mode**: 词向量模式，默认word2vec。可选bert。

- 返回值

  - **w2v模式**

    - 成功：list
    - 失败：-1(词表中没有对应词)

  - **bert模式**

    tuple，(全部输入的编码,cls位置对应编码)

    **np_sen**: np-array, batch_size * seq_length * hidden_size

    **np_cls**: np-arrau,batch_size * hidden_size

- 示例

  ```python
  word = '微信'
  sentence = '我来到清华大学'
  word_vector = demo.get_vector(word)
  result = demo.get_vector(sentence,mode='bert')
  ```

- 参考文献

  - [使用gensim训练词向量](https://radimrehurek.com/gensim/models/word2vec.html)
  - [transformers-BertModel](https://huggingface.co/transformers/model_doc/bert.html?highlight=bertmodel#transformers.BertModel)
	- [bert](https://arxiv.org/pdf/1810.04805.pdf)
  
### 2.4 相似度计算

计算两个词的相似度

- 方法名: compute_similarity()

- 参数

  - **word1**: 字符,待计算的词
  - **word2**: 字符,待计算的词

- 返回值

  - **成功**：0~1之间的浮点数。
  - **失败**：-1。待计算词有词不在词表中。

- 示例

  ```python
  w1 = '国王'
  w2 = '皇后'
  score = demo.compute_similarity(w1,w2)
  ```

### 2.5 相似词查找

返回输入词组的相似词

- 方法名: find_most_similar_words()

- 参数

  - **word**: 格式 str或[str],待查找的种子词或多个种子词
  - **K**: 返回的相似词数量

- 返回值

  - **成功**：[(word1,similarity),(word2,similarity)]
  - **失败**：[ ], 输入的词不在词表中

- 示例

  ```python
  word = "国王"
  words_group = ["国王","皇后","首相"]
  result1 = demo.find_most_similar_words(word)
  result2 = demo.find_most_similar_words(words_group)
  
  ```

  ### 2.6 新词发现

基于给定语料，进行新词发现

- 方法名：find_new_words()

- 参数

  - **inputs**: txt文本文件路径或[str,str...]
  - **words_num**: 返回的新词数量,默认200

- 返回值

  [word1,word2,...wordn]

- 示例

  ```python
  input_file = '~/corpus.txt'
  new_words = demo.find_new_words(input_file)
  ```

- 参考文献

  - [新词发现](https://zhuanlan.zhihu.com/p/80385615)

### 2.7 词云

基于给定语料生产词云

- 方法名：create_word_cloud()

- 参数

  - **inputs**: 输入语料,建议分词并过滤
  - **outputs**: 图片保存路径
  - **bg_img**:  背景图片路径, 默认为矩形
  - **color**: 背景颜色，默认黑色

- 示例

  ```python
  demo.create_word_cloud(inputs=input_file,output_path=None,bg_img=None)
  ```

- 参考文献

  - [wordcloud](https://amueller.github.io/word_cloud/index.html)
  - [中文词云实例](https://github.com/TommyZihao/zihaowordcloud)



### 2.8 加载diy词向量

**方法1：**

修改```config.py```中```WORD2VEC_MODEL_PATH```路径

模型中的word2vec模型会替换为新的词向量

**方法2：**

调用加载函数，加载新的模型

```python
new_model_path = '~'
demo.load_w2v(model_path=new_model_path)
```

