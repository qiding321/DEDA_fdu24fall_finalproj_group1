#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: Ding
Time: 2024/10/23 16:44
"""

import os
import sys
import pandas as pd
import datetime

from my_logger import this_log

p_in = r'E:\Document\MyStrategy\DEDA_fdu24fall_finalproj_group1\final\output' + '\\'
p_out = r'E:\Document\MyStrategy\DEDA_fdu24fall_finalproj_group1\final\output' + '\\'
p_index = p_in+'data_index.pkl'
p_minute = p_in+'data_minute.pkl'


def _get_trading_date(s, d_l):
    d = s[:8]
    hms = s[8:14]
    if hms >= '085000':
        td = [_ for _ in d_l if _ > d]
    else:
        td = [_ for _ in d_l if _ >= d]
    if td:
        return td[0]
    else:
        return None


def _get_preday(x, l):
    if x is None:
        return None
    d_l = [_ for _ in l if _ < x]
    if d_l:
        return d_l[-1]
    else:
        return None


def _get_preweek(x, l):
    if x is None:
        return None
    dt = datetime.datetime.strptime(x, '%Y%m%d')
    week_before = dt - datetime.timedelta(days=7)
    d_before = week_before.strftime('%Y%m%d')
    d_l = [_ for _ in l if _ <= d_before]
    if d_l:
        return d_l[-1]
    else:
        return None


def main():
    data_index = pd.read_pickle(p_index)
    data_minute = pd.read_pickle(p_minute)
    date_list = sorted(data_minute['date'].unique())
    data_index = data_index.sort_values('time')
    data_index['sanhuozujin_zonghe'] = data_index['sanhuozujin_zonghe'].astype(float)
    cols_index = [_ for _ in data_index.columns if _ not in ['time', 'date', 'preday', 'preweek']]
    data_index['date'] = data_index['time'].apply(lambda x: _get_trading_date(x, date_list))
    data_index['preday'] = data_index['date'].apply(lambda x: _get_preday(x, date_list))
    data_index['preweek'] = data_index['date'].apply(lambda x: _get_preweek(x, date_list))

    data_index_nodup = data_index.drop_duplicates('date', keep='last')
    data_index_nodup = data_index_nodup[pd.notnull(data_index_nodup[['date', 'preday', 'preweek']]).all(axis=1)]
    data_index_preday = data_index_nodup.set_index('date').reindex(data_index_nodup['preday'].values)
    data_index_preday.index = data_index_nodup.index
    data_index_preweek = data_index_nodup.set_index('date').reindex(data_index_nodup['preweek'].values)
    data_index_preweek.index = data_index_nodup.index

    cols_factor = []
    for col in cols_index:
        data_index_nodup[col+'_pct_1d'] = data_index_nodup[col]/data_index_preday[col] - 1
        data_index_nodup[col+'_pct_1w'] = data_index_nodup[col]/data_index_preweek[col] - 1
        cols_factor.append(col)
        cols_factor.append(col+'_pct_1d')
        cols_factor.append(col+'_pct_1w')

    idx = data_minute['date'].values
    data_index_to_merge = data_index_nodup.set_index('date').reindex(idx)[cols_factor]
    for col in cols_factor:
        data_minute[col] = data_index_to_merge[col].values

    data_minute.to_pickle(p_out+'merged_minute_data.pkl')
    this_log.info('save data {} to {}'.format(data_minute.shape, p_out+'merged_minute_data.pkl'))
    with open(p_out+'index_factor_name.txt', 'w') as f:
        f.write(','.join(cols_factor))
    this_log.info('save {} factor names to {}'.format(len(cols_factor), p_out+'index_factor_name.txt'))


if __name__ == "__main__":
    main()
