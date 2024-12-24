from sys import path
from numpy.core.fromnumeric import size
import pandas as pd
import numpy as np
import math

dic = {'0': 'athletics', '1': 'cricket', '2': 'football', '3': 'rugby', '4': 'tennis'}

df_mtx = pd.read_table('./bbcsport/bbcsportmtx.txt', sep=' ', header=None, index_col=None, skiprows=2)
print(df_mtx)
# print(df_mtx.dtypes)
# df_mtx = df_mtx.astype(int)
# print(df_mtx[:5])
# print(df_mtx.dtypes)

df_classes = pd.read_table('./bbcsport/bbcsportclasses.txt', sep=' ', header=None, index_col=0, skiprows=4, names=[0])
print(df_classes)

df_terms = pd.read_table('./bbcsport/bbcsportterms.txt', header=None)
df_terms.index = range(1, len(df_terms)+1)
print(df_terms[0:5])


doc_range = list(range(1, 738))
term_range = list(range(1, 4615))
VSM = pd.DataFrame(index=doc_range, columns=term_range)
# 计划用行向量储存每篇文章的特征向量
# print(VSM)


def word_count(file_path):
    with open(file_path, 'r', encoding='gb18030', errors='ignore') as f:
        content = f.read()
        # print(content,type(content))
        n = content.count(' ')+math.ceil(content.count('\n')/2)
        f.close()
        return n
# word_count统计文本中的单词总数(采用的是数' '的方式,一般与实际的词数差距不会太大,并且简单明了)


path_prefix = './bbcsport_raw/'
# print(word_count(path_prefix+dic['0']+'\\001.txt'))


def TF_IDF(doc_index, file_path, term):
    a = df_terms[(df_terms[0] == term)].index.values[0]
    b = word_count(file_path)
    e = df_mtx[(df_mtx[0] == a) & (df_mtx[1] == doc_index)].index.values
    if e.size > 0:
        c = df_mtx[2][df_mtx[(df_mtx[0] == a) & (df_mtx[1] == doc_index)].index.values[0]]
    else:
        return 0
    TF = c/b
    # return TF
    d = len((df_mtx[df_mtx[0] == a]).index.values)+1
    IDF = math.log(737/d)
    # return IDF
    tf_idf = TF*IDF
    return tf_idf
# print(TF_IDF(1,path_prefix+dic['0']+'\\001.txt','hunt'))
# TF_IDF用于计算某一term在某一document中的TF-IDF值


VSM.at[1, 1] = TF_IDF(1, path_prefix+dic['1']+'\\'+str(102-101).zfill(3)+'.txt', df_terms.at[1, 0])
print(VSM)

for i in range(1, 738):
    for j in range(1, 4614):
        if i <= 101:
            VSM.at[i, j] = TF_IDF(i, path_prefix+dic['0']+'\\'+str(i).zfill(3)+'.txt', df_terms.at[j, 0])
            VSM.at[i, 4614] = 0
            print(i, j)
        elif i <= 225:
            VSM.at[i, j] = TF_IDF(i, path_prefix+dic['1']+'\\'+str(i-101).zfill(3)+'.txt', df_terms.at[j, 0])
            VSM.at[i, 4614] = 1
            print(i, j)
        elif i <= 490:
            VSM.at[i, j] = TF_IDF(i, path_prefix+dic['2']+'\\'+str(i-225).zfill(3)+'.txt', df_terms.at[j, 0])
            VSM.at[i, 4614] = 2
            print(i, j)
        elif i <= 637:
            VSM.at[i, j] = TF_IDF(i, path_prefix+dic['3']+'\\'+str(i-490).zfill(3)+'.txt', df_terms.at[j, 0])
            VSM.at[i, 4614] = 3
            print(i, j)
        else:
            VSM.at[i, j] = TF_IDF(i, path_prefix+dic['4']+'\\'+str(i-637).zfill(3)+'.txt', df_terms.at[j, 0])
            VSM.at[i, 4614] = 4
            print(i, j)
VSM.to_csv('C:/Users/wsq/Desktop/ML_EXP/BBC/VSM.csv')
# 双循环的64位浮点运算很慢,慢到难以忍受(全程大概90分钟);事后舍友说采用pandas自带的transform方法或许会变快,有兴趣的话可以尝试一下.



