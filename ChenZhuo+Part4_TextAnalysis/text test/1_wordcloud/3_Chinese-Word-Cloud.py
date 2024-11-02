import os
import re
import wordcloud
import matplotlib.pyplot as plt
from wordcloud import WordCloud,ImageColorGenerator
import numpy as np
import jieba.analyse
from collections import Counter
plt.rcParams['font.sans-serif'] = [u'SimHei']
plt.rcParams['axes.unicode_minus'] = False
import jieba.posseg as pseg
import PIL.Image as Image
from matplotlib import colors


 
# 阅读文本（这里yourfile.txt，根据文本所在具体位置进行设置）
text = open("text test/output/内容_欧洲航线.txt", encoding="utf-8").read()
words = pseg.cut(text)
 
# 按指定长度和词性提取词
report_words = []
for word, flag in words:
    if (len(word) >= 2) and ('n' in flag): #这里设置统计的字数
        report_words.append(word)
 
# 统计高频词汇
result = Counter(report_words).most_common(200) #词的个数
 
# 建立词汇字典
content = dict(result)
#输出词频统计结果
for i in range(50):
    word,flag=result[i]
    print("{0:<10}{1:>5}".format(word,flag))
 
# 设置停用词
stopwords = set()
temp_content = [line.strip() for line in open(r'text test/1_wordcloud/baidu_stopwords.txt','r', encoding='UTF-8').readlines()]
stopwords.update(temp_content)
 

mask = np.array(Image.open("text test/1_wordcloud/ship.jpg"))
'''
# 如果当前位深是32的话，可以不用写转RGBA模式的这一句，但是写上也没啥问题
# 从RGB（24位）模式转成RGBA（32位）模式
img = Image.open("yourfile.png").convert('RGBA')
W, L = img.size
white_pixel = (0, 0, 0, 0)  # 白色
for h in range(W):
    for i in range(L):
        if img.getpixel((h, i)) == white_pixel:
            img.putpixel((h, i), (255, 255, 255, 0))  # 设置透明
img.save("yourfile_new.png")  # 自己设置保存地址
'''

 
# 生成词云
image_colors=ImageColorGenerator(mask)
wordcloud=WordCloud(background_color=None,mode="RGBA",width=1000,height=1000,color_func=image_colors ,
                 mask=mask,font_path='text test/1_wordcloud/font2.ttf',stopwords=stopwords,scale=4,
                 collocations=False,min_font_size=10,max_font_size=1000)
wordcloud.generate_from_frequencies(content)

# 使用 matplotlib 显示词云
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
# 保存词云图
wordcloud.to_file("text test/output/wordcloud_欧洲航线.png")