#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: Ding
Time: 2024/10/23 16:44
"""

import os
import sys
import numpy as np
import pandas as pd
import tqdm
from sklearn.linear_model import LinearRegression, BayesianRidge

from my_logger import this_log

p_min_data = r'G:\data\future_data\tick_data' + '\\'
p_daily_data = r'E:\Document\MyStrategy\DEDA_fdu24fall_finalproj_group1\data\ec_fut_daily.csv'
p_date_list = r'G:\data\date_list\date_list.txt'
DATE_LIST_ALL = sorted(set(open(p_date_list).read().split('\n')))
p_out = r'E:\Document\MyStrategy\DEDA_fdu24fall_finalproj_group1\final\output'+'\\'


def get_date_diff(d1, d2, d_list):
    d11 = [_ for _ in d_list if _ >= d1][0]
    d21 = [_ for _ in d_list if _ <= d2][-1]
    return d_list.index(d21)-d_list.index(d11)

def get_r2(pv, rv, pv_benchmark):
    ss_res = ((rv - pv) ** 2).sum()  # sum of squares of residuals
    ss_tot = ((rv - pv_benchmark) ** 2).sum()  # total sum of squares
    r2 = 1 - (ss_res / ss_tot)  # R-squared formula
    return r2


def get_data_daily():
    daily_data_raw = pd.read_csv(p_daily_data)
    daily_main_con = daily_data_raw[daily_data_raw['MAINCON'] == 1]
    data_daily = daily_main_con[[
        'TRADE_DATE', 'TICKER_SYMBOL', 'PRE_CLOSE_PRICE',
        'PRE_SETTL_PRICE', 'OPEN_PRICE', 'LAST_TRADE_DATE',
        'CLOSE_PRICE',
        'OPEN_INT'
    ]]
    data_daily['TRADE_DATE'] = data_daily['TRADE_DATE'].apply(lambda x: x.replace('-', ''))
    data_daily['LAST_TRADE_DATE'] = data_daily['LAST_TRADE_DATE'].apply(lambda x: x.replace('-', ''))
    data_daily['DAYS_TO_MAT'] = [get_date_diff(_[0], _[1], DATE_LIST_ALL) for _ in zip(data_daily['TRADE_DATE'], data_daily['LAST_TRADE_DATE'])]
    assert len(data_daily['TRADE_DATE'].unique()) == len(data_daily), 'TRADE_DATE not unique'
    return data_daily


def get_minute_data_one_day(date, contract):
    p = p_min_data+date+'/'+contract.upper()+'.INE.pkl'
    if not os.path.exists(p):
        this_log.info('file not exists: {}'.format(p))
        return
    data = pd.read_pickle(p)

    def _time_to_sec(t):
        h = t.hour
        m = t.minute
        s = t.second
        mi = t.microsecond
        sec = 3600*(h-9)+60*m+s+mi/1000000
        if sec >= 16200:
            sec -= (7200+900)
        elif sec >= 5400:
            sec -= 900
        return sec

    data['sec'] = data['time'].apply(_time_to_sec)
    data['min'] = data['sec']//60
    data['mid'] = (data['bid1']+data['ask1'])/2
    data = data[pd.notnull(data['mid'])]
    data_last = data.groupby('min').last()

    def _agg(g):
        f = g.iloc[0, :]
        l = g.iloc[-1, :]
        idx_max = g['mid'].idxmax()
        idx_min = g['mid'].idxmin()
        max_time = g.loc[idx_max, 'sec']
        min_time = g.loc[idx_min, 'sec']
        duration = l['sec']-f['sec']
        max_time_pos = (max_time-f['sec'])/duration if duration != 0 else 0
        min_time_pos = (min_time-f['sec'])/duration if duration != 0 else 0
        cum_ret = g['mid']/f['mid']-1
        std = cum_ret.std()
        if len(cum_ret) >= 5:
            lr = LinearRegression()
            x = np.arange(len(cum_ret)).reshape(-1, 1)
            lr.fit(x, cum_ret)
            slope = lr.coef_[0]
            intercept = lr.intercept_
            r2 = get_r2(lr.predict(x), cum_ret.values, cum_ret.mean())
        else:
            slope, intercept, r2 = 0, 0, 0

        return pd.Series({
            'o_m': f['mid'],
            'h_m': g['mid'].max(),
            'l_m': g['mid'].min(),
            'c_m': l['mid'],
            'n_m': len(g),
            'max_time_pos_m': max_time_pos,
            'min_time_pos_m': min_time_pos,
            'std_m': std,
            'slope_m': slope,
            'intercept_m': intercept,
            'r2_m': r2,
        })

    data_agg = data.groupby('min').apply(_agg)
    data_min = pd.concat([data_last, data_agg], axis=1)
    data_min['cumvolume'] = data_min['volume']
    data_min['cumamount'] = data_min['amount']
    data_min['volume'] = data_min['cumvolume'].diff().fillna(0)
    data_min['amount'] = data_min['cumamount'].diff().fillna(0)
    data_min['position_chg'] = data_min['position'].diff().fillna(0)
    this_log.info('get minute data {} from {}'.format(data_min.shape, p))
    return data_min


def get_data_minute_main_contract(dates, data_daily):
    data_list = []
    for date in tqdm.tqdm(dates):
        data_daily_one_day_ = data_daily[data_daily['TRADE_DATE'] == date]
        assert len(data_daily_one_day_) == 1
        data_daily_one_day = data_daily_one_day_.iloc[0]
        contract = data_daily_one_day['TICKER_SYMBOL']
        minute_data = get_minute_data_one_day(date, contract)
        if minute_data is None:
            continue
        minute_data['date'] = data_daily_one_day['TRADE_DATE']
        minute_data['symbol'] = data_daily_one_day['TICKER_SYMBOL']
        minute_data['pre_close'] = data_daily_one_day['PRE_CLOSE_PRICE']
        minute_data['pre_settle'] = data_daily_one_day['PRE_SETTL_PRICE']
        minute_data['open'] = data_daily_one_day['OPEN_PRICE']
        minute_data['last_day'] = data_daily_one_day['LAST_TRADE_DATE']
        minute_data['day_to_mat'] = data_daily_one_day['DAYS_TO_MAT']
        minute_data['close'] = data_daily_one_day['CLOSE_PRICE']
        minute_data['open_interest'] = data_daily_one_day['OPEN_INT']
        data_list.append(minute_data)
    data_df = pd.concat(data_list, ignore_index=True)
    return data_df


def main():
    date1 = '20230818'
    date2 = '20240927'
    data_daily = get_data_daily()
    dates = sorted(set(data_daily['TRADE_DATE'].unique()))
    data_minute = get_data_minute_main_contract(dates, data_daily)
    p_out_minute = p_out+'data_minute.pkl'
    pd.to_pickle(data_minute, p_out_minute)
    this_log.info('save minute data {} to {}'.format(data_minute.shape, p_out_minute))


if __name__ == "__main__":
    main()
