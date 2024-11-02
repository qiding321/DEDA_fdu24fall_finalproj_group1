#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: Ding
Time: 2024/10/24 20:03
"""

import os
import sys
import pandas as pd


import os
import sys
import pandas as pd
import datetime

from my_logger import this_log


p_in = r'E:\Document\MyStrategy\DEDA_fdu24fall_finalproj_group1\final\output' + '\\'
p_out = r'E:\Document\MyStrategy\DEDA_fdu24fall_finalproj_group1\final\output' + '\\'
p_merged = p_out+'merged_minute_data.pkl'

p_sentiment_euro1 = r'E:\Document\MyStrategy\DEDA_fdu24fall_finalproj_group1\weibo-public-opinion-analysis\output\评论数据\欧洲航线_评论爬取用URL.xlsx'
p_sentiment_redsea1 = r'E:\Document\MyStrategy\DEDA_fdu24fall_finalproj_group1\weibo-public-opinion-analysis\output\评论数据\红海危机_评论爬取用URL.xlsx'
p_sentiment_index1 = r'E:\Document\MyStrategy\DEDA_fdu24fall_finalproj_group1\weibo-public-opinion-analysis\output\评论数据\集运指数_评论爬取用URL.xlsx'
p_sentiment_euro2 = r'E:\Document\MyStrategy\DEDA_fdu24fall_finalproj_group1\weibo-public-opinion-analysis\output\终-欧洲航线评论数据全.xlsx'
p_sentiment_redsea2 = r'E:\Document\MyStrategy\DEDA_fdu24fall_finalproj_group1\weibo-public-opinion-analysis\output\终-红海危机评论数据全.xlsx'
p_sentiment_index2 = r'E:\Document\MyStrategy\DEDA_fdu24fall_finalproj_group1\weibo-public-opinion-analysis\output\终-集运指数评论数据全.xlsx'


def get_sentiment(p):
    data = pd.read_excel(p)
    if '发文时间' in data.columns:
        time_idx = '发文时间'
        time_format = 'short'
    elif '发布时间' in data.columns:
        time_idx = '发布时间'
        time_format = 'short'
    elif '评论时间' in data.columns:
        time_idx = '评论时间'
        time_format = 'long'
    else:
        raise ValueError

    if time_format == 'short':
        t = data[time_idx].apply(lambda x: datetime.datetime.strptime(x, '%y-%m-%d %H:%M'))
    elif time_format == 'long':
        t = pd.to_datetime(data[time_idx])
    else:
        raise ValueError
    data2 = pd.DataFrame({'time': t, 'sentiment0': data['sentiment'].values})
    data2 = data2.dropna()
    data2['sentiment'] = (data2['sentiment0'] == 'positive').astype(int)
    data2['date'] = data2['time'].apply(lambda x: x.strftime('%Y%m%d'))
    return data2

def main():
    senti_euro = pd.concat([get_sentiment(p_sentiment_euro1), get_sentiment(p_sentiment_euro2)], ignore_index=True)
    senti_redsea = pd.concat([get_sentiment(p_sentiment_redsea1), get_sentiment(p_sentiment_redsea2)], ignore_index=True)
    senti_index = pd.concat([get_sentiment(p_sentiment_index1), get_sentiment(p_sentiment_index2)], ignore_index=True)
    d_raw = {
        'euro': senti_euro,
        'redsea': senti_redsea,
        'index': senti_index
    }
    d_out_list = list()
    for name, d in d_raw.items():
        pos_by_date = d.groupby('date')['sentiment'].sum()
        len_by_date = d.groupby('date')['sentiment'].count()
        pct_by_date = pos_by_date/len_by_date-0.5
        d_out = pd.DataFrame({
            'sentiment'+'_'+name+'_'+'pos_num': pos_by_date,
            'sentiment'+'_'+name+'_'+'tot_num': len_by_date,
            'sentiment'+'_'+name+'_'+'pos_pct': pct_by_date,
        })
        d_out_list.append(d_out)
    d_out_df = pd.concat(d_out_list, axis=1)
    d_out_df.to_pickle(p_out+'sentiment_by_date.pkl')
    this_log.info('save sentiment {} to {}'.format(d_out_df.shape, p_out+'sentiment_by_date.pkl'))

    data_minute = pd.read_pickle(p_merged)
    date_list_all = [_.strftime('%Y%m%d') for _ in pd.date_range('20230801', '20241031')]
    dates = sorted(data_minute['date'].unique())

    def _rename(data, suffix):
        data = data.rename(index={_: _+suffix for _ in data.index})
        return data

    d_record = dict()
    for d in dates:
        idx = date_list_all.index(d)
        d_1 = date_list_all[idx-1]
        d_3 = [date_list_all[idx-1], date_list_all[idx-2], date_list_all[idx-3]]
        d_1_data = _rename(d_out_df.reindex([d_1]).mean(), '_d1')
        d_3_data = _rename(d_out_df.reindex(d_3).mean(), '_d3')
        d_record[d] = pd.concat([d_1_data, d_3_data])
    d_df = pd.DataFrame(d_record).T
    d_df2 = d_df.reindex(data_minute['date'].values)
    for col in d_df2.columns:
        data_minute[col] = d_df2[col].values
    pd.to_pickle(data_minute, p_out+'merged_minute_data_with_sentiment.pkl')
    this_log.info('save {} data to {}'.format(data_minute.shape, p_out+'merged_minute_data_with_sentiment.pkl'))

    with open(p_out+'sentiment_factor_name.txt', 'w') as f:
        f.write(','.join(d_df2.columns))



if __name__ == "__main__":
    main()
