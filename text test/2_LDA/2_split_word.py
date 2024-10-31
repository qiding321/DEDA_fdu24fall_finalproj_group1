import os
import re
import PIL
import wordcloud
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
import jieba.analyse
from collections import Counter
plt.rcParams['font.sans-serif'] = [u'SimHei']
plt.rcParams['axes.unicode_minus'] = False



def stopwordslist():
    stopwords = [line.strip() for line in open('text test/1_wordcloud/baidu_stopwords.txt','r', encoding='utf-8').readlines()]
    return stopwords


words_str = ""
with open("text test/content data/cleaning_红海危机.txt",'r',encoding = 'utf-8') as f:
    for line in f:
        line = re.sub(u"[0-9\s+.!/,$%^*()?;；:-【】+\"\']+|[+——！，;:。？、~@#￥%……&*（）＞＜-]+", "", line)  # 去掉多余字符
        if line == "":continue
        line = line.replace("\n", "") # 去掉换行符
        seg_list = jieba.cut(line, cut_all=False)
        words_str += (" ".join(seg_list))
stopwords = stopwordslist()
words = [word for word in words_str.split(" ") if word not in stopwords and len(word) > 1] # 去除停用词和去除单个文字
f=open('text test/content data/splitword_红海危机.txt','w',encoding='utf-8')
f.write(str(words))
f.close()