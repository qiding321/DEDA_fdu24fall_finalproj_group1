#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: Ding
Time: 2024/10/23 16:44
"""

import os
import sys
import pandas as pd
import numpy as np
from my_logger import this_log


p_in = r'E:\Document\MyStrategy\DEDA_fdu24fall_finalproj_group1\final\output' + '\\'
p_out = r'E:\Document\MyStrategy\DEDA_fdu24fall_finalproj_group1\final\output' + '\\'
p_merged = p_in+'merged_minute_data_with_sentiment.pkl'
p_factor_name_index = p_in+'index_factor_name.txt'
p_factor_name_sentiment = p_in+'sentiment_factor_name.txt'


def _get_pos(ts, time_ts, window, min_or_max):
    res = []
    for i in range(len(ts)):
        sub_ts = ts.iloc[max(i-window, 0):i+1].values
        sub_time_ts = time_ts.iloc[max(i-window, 0):i+1].values
        idx = (np.argmax if min_or_max == 'max' else np.argmin)(sub_ts)
        pos_within_1min = sub_time_ts[idx]
        res.append((idx+pos_within_1min)/window)
    return pd.Series(res, index=ts.index)


def main():
    data_raw = pd.read_pickle(p_merged)
    factor_name_raw_index = open(p_factor_name_index).read().split(',')
    factor_name_raw_sentiment = open(p_factor_name_sentiment).read().split(',')

    data_raw = data_raw.rename(columns={ab+'size'+str(l): ab+'s'+str(l) for l in range(1, 6) for ab in ['b', 'a']})
    factor_name_raw_hf = [_ for _ in data_raw.columns if _ not in factor_name_raw_index and _ not in factor_name_raw_sentiment]
    nonfactor_columns = [
        'time', 'code', 'date', 'symbol', 'last_day', 'close', 'open_interest'
    ]

    data_raw['spread'] = (data_raw['ask1']-data_raw['bid1'])/data_raw['mid']
    data_raw['ret_1min'] = data_raw['mid'].diff()/data_raw['mid']
    data_raw['ret_5min'] = data_raw['ret_1min'].rolling(window=5).mean()
    data_raw['ret_10min'] = data_raw['ret_1min'].rolling(window=10).mean()
    data_raw['ret_30min'] = data_raw['ret_1min'].rolling(window=30).mean()
    data_raw['pos_1min'] = (data_raw['mid']-data_raw['l_m'])/(data_raw['h_m']-data_raw['l_m'])
    data_raw['o_5min'] = data_raw['o_m'].shift(5)
    data_raw['o_10min'] = data_raw['o_m'].shift(10)
    data_raw['o_30min'] = data_raw['o_m'].shift(30)
    data_raw['h_5min'] = data_raw['h_m'].rolling(window=5).max()
    data_raw['h_10min'] = data_raw['h_m'].rolling(window=10).max()
    data_raw['h_30min'] = data_raw['h_m'].rolling(window=30).max()
    data_raw['l_5min'] = data_raw['l_m'].rolling(window=5).min()
    data_raw['l_10min'] = data_raw['l_m'].rolling(window=10).min()
    data_raw['l_30min'] = data_raw['l_m'].rolling(window=30).min()
    data_raw['h_pos_5min'] = _get_pos(data_raw['h_m'], data_raw['max_time_pos_m'], 5, 'max')
    data_raw['h_pos_10min'] = _get_pos(data_raw['h_m'], data_raw['max_time_pos_m'], 10, 'max')
    data_raw['h_pos_30min'] = _get_pos(data_raw['h_m'], data_raw['max_time_pos_m'], 30, 'max')
    data_raw['l_pos_5min'] = _get_pos(data_raw['l_m'], data_raw['min_time_pos_m'], 5, 'min')
    data_raw['l_pos_10min'] = _get_pos(data_raw['l_m'], data_raw['min_time_pos_m'], 10, 'min')
    data_raw['l_pos_30min'] = _get_pos(data_raw['l_m'], data_raw['min_time_pos_m'], 30, 'min')
    data_raw['hl_1min'] = (data_raw['h_m']-data_raw['l_m'])/data_raw['o_m']
    data_raw['hl_5min'] = (data_raw['h_5min']-data_raw['l_5min'])/data_raw['o_5min']
    data_raw['hl_10min'] = (data_raw['h_10min']-data_raw['l_10min'])/data_raw['o_10min']
    data_raw['hl_30min'] = (data_raw['h_30min']-data_raw['l_30min'])/data_raw['o_30min']
    data_raw['mid_lastminmax_1min'] = np.where(data_raw['max_time_pos_m']>data_raw['min_time_pos_m'], data_raw['h_m'], data_raw['l_m'])
    data_raw['mid_lastminmax_5min'] = np.where(data_raw['h_pos_5min']>data_raw['l_pos_5min'], data_raw['h_5min'], data_raw['l_5min'])
    data_raw['mid_lastminmax_10min'] = np.where(data_raw['h_pos_10min']>data_raw['l_pos_10min'], data_raw['h_10min'], data_raw['l_10min'])
    data_raw['mid_lastminmax_30min'] = np.where(data_raw['h_pos_30min']>data_raw['l_pos_30min'], data_raw['h_30min'], data_raw['l_30min'])
    data_raw['mid_firstminmax_1min'] = np.where(data_raw['max_time_pos_m']<data_raw['min_time_pos_m'], data_raw['h_m'], data_raw['l_m'])
    data_raw['mid_firstminmax_5min'] = np.where(data_raw['h_pos_5min']<data_raw['l_pos_5min'], data_raw['h_5min'], data_raw['l_5min'])
    data_raw['mid_firstminmax_10min'] = np.where(data_raw['h_pos_10min']<data_raw['l_pos_10min'], data_raw['h_10min'], data_raw['l_10min'])
    data_raw['mid_firstminmax_30min'] = np.where(data_raw['h_pos_30min']<data_raw['l_pos_30min'], data_raw['h_30min'], data_raw['l_30min'])
    data_raw['ret_pos_1min'] = data_raw['mid']/data_raw['mid_lastminmax_1min']-1
    data_raw['ret_pos_5min'] = data_raw['mid']/data_raw['mid_lastminmax_5min']-1
    data_raw['ret_pos_10min'] = data_raw['mid']/data_raw['mid_lastminmax_10min']-1
    data_raw['ret_pos_30min'] = data_raw['mid']/data_raw['mid_lastminmax_30min']-1
    data_raw['ret_pos_1min_first'] = data_raw['mid']/data_raw['mid_firstminmax_1min']-1
    data_raw['ret_pos_5min_first'] = data_raw['mid']/data_raw['mid_firstminmax_5min']-1
    data_raw['ret_pos_10min_first'] = data_raw['mid']/data_raw['mid_firstminmax_10min']-1
    data_raw['ret_pos_30min_first'] = data_raw['mid']/data_raw['mid_firstminmax_30min']-1

    data_raw['pos_5min'] = (data_raw['mid']-data_raw['l_5min'])/(data_raw['h_5min']-data_raw['l_5min'])
    data_raw['pos_10min'] = (data_raw['mid']-data_raw['l_10min'])/(data_raw['h_10min']-data_raw['l_10min'])
    data_raw['pos_30min'] = (data_raw['mid']-data_raw['l_30min'])/(data_raw['h_30min']-data_raw['l_30min'])


    def _get_oib(b, a):
        return (b-a)/(b+a)

    data_raw['bs3_sum'] = data_raw['bs1']+data_raw['bs2']+data_raw['bs3']
    data_raw['as3_sum'] = data_raw['as1']+data_raw['as2']+data_raw['as3']
    data_raw['bs5_sum'] = data_raw['bs1']+data_raw['bs2']+data_raw['bs3']+data_raw['bs4']+data_raw['bs5']
    data_raw['as5_sum'] = data_raw['as1']+data_raw['as2']+data_raw['as3']+data_raw['as4']+data_raw['as5']
    for w in [5, 10]:
        data_raw[f'bs1_{w}min'] = data_raw['bs1'].rolling(window=w).mean()
        data_raw[f'as1_{w}min'] = data_raw['as1'].rolling(window=w).mean()
        data_raw[f'bs3_sum_{w}min'] = data_raw['bs3_sum'].rolling(window=w).mean()
        data_raw[f'as3_sum_{w}min'] = data_raw['as3_sum'].rolling(window=w).mean()
        data_raw[f'bs5_sum_{w}min'] = data_raw['bs5_sum'].rolling(window=w).mean()
        data_raw[f'as5_sum_{w}min'] = data_raw['as5_sum'].rolling(window=w).mean()

    data_raw['oib1_1min'] = _get_oib(data_raw['bs1'],data_raw['as1'])
    data_raw['oib3_1min'] = _get_oib(data_raw['bs3_sum'], data_raw['as3_sum'])
    data_raw['oib5_1min'] = _get_oib(data_raw['bs5_sum'], data_raw['as5_sum'])
    data_raw['oib1_5min'] = _get_oib(data_raw['bs1_5min'],data_raw['as1_5min'])
    data_raw['oib3_5min'] = _get_oib(data_raw['bs3_sum_5min'], data_raw['as3_sum_5min'])
    data_raw['oib5_5min'] = _get_oib(data_raw['bs5_sum_5min'], data_raw['as5_sum_5min'])
    data_raw['oib1_10min'] = _get_oib(data_raw['bs1_10min'],data_raw['as1_10min'])
    data_raw['oib3_10min'] = _get_oib(data_raw['bs3_sum_10min'], data_raw['as3_sum_10min'])
    data_raw['oib5_10min'] = _get_oib(data_raw['bs5_sum_10min'], data_raw['as5_sum_10min'])

    data_raw['open_interest_chg'] = data_raw['position'].diff()
    data_raw['amt_5min'] = data_raw['amount'].rolling(window=5).mean()
    data_raw['amt_10min'] = data_raw['amount'].rolling(window=10).mean()

    def _get_cross(s1, s2):
        diff = (s1 > s2).astype(float)
        return (diff-diff.shift(1))/2

    data_raw['mid_5min'] = data_raw['mid'].rolling(window=5).mean()
    data_raw['mid_10min'] = data_raw['mid'].rolling(window=10).mean()
    data_raw['mid_cross_1_5'] = _get_cross(data_raw['mid'], data_raw['mid_5min'])
    data_raw['mid_cross_5_10'] = _get_cross(data_raw['mid_5min'], data_raw['mid_10min'])

    factor_col = [_ for _ in data_raw.columns if _ not in nonfactor_columns]
    factor = data_raw.set_index('time')[factor_col]
    pd.to_pickle(factor, p_out+'factor.pkl')
    this_log.info('save {} data to {}'.format(factor.shape, p_out+'factor.pkl'))

    # ================= y =================
    y_dict = dict()
    for t in [1, 5, 10]:
        y_ret = data_raw['mid'].shift(-t)/data_raw['mid']-1
        y_ret_wave = data_raw['ret_pos_1min_first'].shift(-t)
        idx_missing = data_raw['code'] != data_raw['code'].shift(-t)
        y_ret[idx_missing] = np.nan
        y_ret_wave[idx_missing] = np.nan
        y_dict[f'y_ret_{t}m'] = y_ret
        y_dict[f'y_ret_wave_{t}m'] = y_ret_wave

    y_df = pd.DataFrame(y_dict)

    y_df['time'] = data_raw['time']
    target = y_df.set_index('time')
    pd.to_pickle(target, p_out+'target.pkl')
    this_log.info('save {} data to {}'.format(target.shape, p_out+'target.pkl'))



if __name__ == "__main__":
    main()
