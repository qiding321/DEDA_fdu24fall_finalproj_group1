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


#词频统计
def get_plt(data, title):
    x = [i[0] for i in data]
    y = [i[1] for i in data]
    fig, ax = plt.subplots()
    ax.barh(range(len(x)), y, color='gold')
    ax.set_yticks(range(len(x)))
    ax.set_yticklabels(x)
    plt.rcParams['font.sans-serif'] = [u'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.title(title, fontsize=10)
    plt.ylabel("word")
    plt.xlabel("times")
    plt.savefig('text test/output/内容_红海危机.png')
    plt.show()
 
# def _wordcloud(word_counts): # 词云生成
#     mask = np.array(PIL.Image.open("text test/1_wordcloud/ship.jpg"))
#     image_colors = wordcloud.ImageColorGenerator(mask)  # 基于彩色图像生成相应彩色 文字颜色跟随背景图颜色
#     wc = WordCloud(scale=4,background_color=None,mode="RGBA",width=1600,height=900,color_func=image_colors ,
#                       mask=mask,font_path='text test/1_wordcloud/font2.ttf', max_words=200, repeat=False,
#                       collocations=False,min_font_size=10,max_font_size=1000)
#     wc.generate_from_frequencies(word_counts)
    
    

#     wc.recolor(color_func=image_colors)
#     wc.to_file("text test/output/wordcloud_红海危机.png")
#     plt.imshow(wc)
#     plt.axis('off')
#     plt.show()

#分词
words_str = ""
with open("text test/output/内容_红海危机.txt",'r',encoding = 'utf-8') as f:
    for line in f:
        line = re.sub(u"[0-9\s+.!/,$%^*()?;；:-【】+\"\']+|[+——！，;:。？、~@#￥%……&*（）＞＜-]+", "", line)  # 去掉多余字符
        if line == "":continue
        line = line.replace("\n", "") # 去掉换行符
        seg_list = jieba.cut(line, cut_all=False)
        words_str += (" ".join(seg_list))
stopwords = stopwordslist()
words = [word for word in words_str.split(" ") if word not in stopwords and len(word) > 1] # 去除停用词和去除单个文字
print(words)

word_counts = Counter()  # 词频统计
for x in words:
    word_counts[x] += 1
get_plt(word_counts.most_common(30), "词频统计top30")