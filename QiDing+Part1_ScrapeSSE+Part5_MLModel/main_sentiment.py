#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: Ding
Time: 2024/10/18 13:13
"""

import os
import sys
import pandas as pd
sys.path.append('/home/dqi/git/DEDA_fud24fall_finalproj_group1/Senta')
from senta import Senta
import tqdm


def main():
    p_weibo_data = r'/home/dqi/git/DEDA_fdu24fall_finalproj_group1/weibo-search-master/weibo/结果文件' + '/'
    p_lda_file = r'/home/dqi/git/DEDA_fdu24fall_finalproj_group1/weibo-public-opinion-analysis/LDA' + '/'
    p_out = r'/home/dqi/git/DEDA_fdu24fall_finalproj_group1/weibo-public-opinion-analysis/output' + '/'

    my_senta = Senta()

    print(my_senta.get_support_model())  # ["ernie_1.0_skep_large_ch", "ernie_2.0_skep_large_en", "roberta_skep_large_en"]
    print(my_senta.get_support_task())  # ["sentiment_classify", "aspect_sentiment_classify", "extraction"]
    use_cuda = False  # 设置True or False

    my_senta.init_model(model_class="ernie_1.0_skep_large_ch", task="sentiment_classify", use_cuda=use_cuda)
    # texts = ["中山大学是岭南第一学府"]
    # result = my_senta.predict(texts)
    # print(result)

    # my_senta.init_model(model_class="ernie_1.0_skep_large_ch", task="extraction", use_cuda=use_cuda)
    # texts = ["唐 家 三 少 ， 本 名 张 威 。"]
    # result = my_senta.predict(texts, '')
    # print(result)
    # sentence_split = pd.read_pickle(p_out+'sentence_split.pkl')
    sentence = pd.read_pickle(p_out+'sentence.pkl')
    sentiment_list = []
    for s in tqdm.tqdm(sentence):
        result = my_senta.predict(s)
        sentiment_list.append(result)
    pd.to_pickle(sentiment_list, p_out+'sentiment_red_sea.pk')

    p_in = '/home/dqi/git/DEDA_fdu24fall_finalproj_group1/data/'
    file_list = [
        '终-红海危机评论数据全.xlsx', '终-集运指数评论数据全.xlsx', '终-欧洲航线评论数据全.xlsx',
        '评论数据/红海危机_评论爬取用URL.xlsx', '评论数据/集运指数_评论爬取用URL.xlsx', '评论数据/欧洲航线_评论爬取用URL.xlsx'
    ]
    p_list = [p_in+_ for _ in file_list]
    data_list = []
    for p in p_list:
        data_list.append(pd.read_excel(p))

    for p, d in zip(file_list, data_list):
        print(p)
        col = '评论内容' if '评论内容' in d.columns else '微博内容'
        d['sentiment'] = ''
        for idx, s in tqdm.tqdm(list(d[col].items())):
            if isinstance(s, str):
                result = my_senta.predict(s)
                d.loc[idx, 'sentiment'] = result[0][1]
        d.to_excel(p_out+p)


if __name__ == "__main__":
    main()
