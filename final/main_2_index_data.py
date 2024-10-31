#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: Ding
Time: 2024/10/23 16:44
"""
import datetime
import tqdm
import os
import sys
import pandas as pd
import numpy as np
from my_logger import this_log


p_index = r'E:\Document\MyStrategy\DEDA_fdu24fall_finalproj_group1\data\sse_scrapy'+'\\'
p_out = r'E:\Document\MyStrategy\DEDA_fdu24fall_finalproj_group1\final\output'+'\\'

def ret_none_if_not_exist(f):
    def g(p):
        if os.path.exists(p):
            return f(p)
        else:
            print(p, 'not exists')
            return None
    return g

@ret_none_if_not_exist
def get_yidaiyilu_data(p):
    _d = pd.read_csv(p, index_col=0)
    _s = _d['本期（点）']
    return {
        'ydyl_mye': _s['“一带一路”贸易额指数'],
        'ydyl_jzx': _s['“一带一路”集装箱海运量指数'],
        'hszczl_yj': _s['“海上丝绸之路”运价指数'],
    }

@ret_none_if_not_exist
def get_yidaiyilu_haiyun_data(p):
    _d = pd.read_csv(p, index_col=0)
    return {
        'ydyl_yunliang': _d.loc['综合指数', '本期']
    }


@ret_none_if_not_exist
def get_sichouzhilu_data(p2):
    # _d = pd.read_csv(p2, index_col=0, )
    l = open(p2, encoding='utf8').read().split('\n')
    cols = l[0].split(',')
    data = []
    for ll in l[1:]:
        words = ll.split(',')
        if len(words) == len(cols):
            data.append(words)
    df = pd.DataFrame(data, columns=cols).set_index('指数')
    df['本期'] = df['本期'].astype(float)
    return {
        'sczl_jinkou': df.loc['“海上丝绸之路”进口集装箱运价指数', '本期'],
        'sczl_jinkou_eu': df.loc['欧洲航线', '本期'].iloc[0],
        'sczl_chukou': df.loc['“海上丝绸之路”出口集装箱运价指数', '本期'],
        'sczl_chukou_eu': df.loc['欧洲航线', '本期'].iloc[1],
    }


@ret_none_if_not_exist
def get_sh_chukoujiesuan_data(p):
    _d = pd.read_csv(p, index_col=0)
    col = [_ for _ in _d.columns if _.startswith('本期')][0]
    return {
        'chukoujiesuan_eu': _d.loc['欧洲航线（基本港）Europe (Base port)', col],
        'chukoujiesuan_us': _d.loc['美西航线（基本港）USWC (Base port)', col],
    }


@ret_none_if_not_exist
def get_sh_chukouyunjia_data(p):
    _d = pd.read_csv(p, index_col=0)
    col = [_ for _ in _d.columns if _.startswith('本期')][0]
    return {
        'sh_chukouyunjia': _d.loc['综合指数 Comprehensive Index', col]
    }


@ret_none_if_not_exist
def get_dongnanyayunjia_data(p):
    _d = pd.read_csv(p, index_col=0)
    col = [_ for _ in _d.columns if _.startswith('本期')][0]
    return {
        'dongnanyayunjia': _d.loc['综合指数', col]
    }


@ret_none_if_not_exist
def get_zg_chukouyunjia_data(p):
    _d = pd.read_csv(p, index_col=0)
    col = [_ for _ in _d.columns if _.startswith('本期')][0]
    return {
        'zgchukouzonghe': _d.loc['中国出口集装箱运价综合指数', col],
        'japan_chukou': _d.loc['(JAPAN SERVICE)', col],
        'eu_chukou': _d.loc['(EUROPE SERVICE)', col],
        'wc_us_chukou': _d.loc['(W/C AMERICA SERVICE)', col],
        'ec_us_chukou': _d.loc['(E/C AMERICA SERVICE)', col],
        'korea_chukou': _d.loc['(KOREA SERVICE)', col],
        'se_asia_chukou': _d.loc['(SOUTHEAST ASIA SERVICE)', col],
        'mediterranean_chukou': _d.loc['(MEDITERRANEAN SERVICE)', col],
        'aus_chukou': _d.loc['(AUSTRALIA/NEW ZEALAND SERVICE)', col],
        'south_africa_chukou': _d.loc['(SOUTH AFRICA SERVICE)', col],
        'south_america_chukou': _d.loc['(SOUTH AMERICA SERVICE)', col],
        'west_east_africa_chukou': _d.loc['(WEST EAST AFRICA SERVICE)', col],
        'persian_chukou': _d.loc['(PERSIAN GULF/RED SEA SERVICE)', col],
    }


@ret_none_if_not_exist
def get_chengpinyou_data(p):
    _d = pd.read_csv(p, index_col=0)
    col = [_ for _ in _d.columns if _.startswith('指数')][0]
    return {
        'chengpinyou_zonghe': _d.loc['综合指数', col],
        'chengpinyou_peisong': _d.loc['配送运价指数', col],
        'chengpinyou_shichangyunjia': _d.loc['市场运价指数', col],
    }

@ret_none_if_not_exist
def get_sanhuoyunjia_data(p):
    _d = pd.read_csv(p, index_col=0)
    col = [_ for _ in _d.columns if _.startswith('本期')][0]
    return {
        'sanhuo_yunjia': _d.loc['综合指数', col],
    }

@ret_none_if_not_exist
def get_meitanyunjia_data(p):
    _d = pd.read_csv(p, index_col=0)
    col = [_ for _ in _d.columns if _.startswith('本期')][0]
    return {
        'meitan_yunjia': _d.loc['综合指数', col]
    }
@ret_none_if_not_exist
def get_liangshiyunjia_data(p):
    _d = pd.read_csv(p, index_col=0)
    # col = [_ for _ in _d.columns if _.startswith('市场运价')][0]
    return {
        'liangshi_yunjia': _d.loc['综合指数', :].iloc[2],
    }
@ret_none_if_not_exist
def get_jinshukuangshi_data(p):
    _d = pd.read_csv(p, index_col=0)
    # col = [_ for _ in _d.columns if _.startswith('市场运价')][0]
    return {
        'jinshukuangshi_yunjia': _d.loc['综合指数', :].iloc[1],
    }
@ret_none_if_not_exist
def get_jinkouyuanyou_data(p):
    _d = pd.read_csv(p, index_col=0)
    col = [_ for _ in _d.columns if _.startswith('本期')][0]
    return {
        'jinkouyuanyou': _d.loc['综合指数', col],
    }
@ret_none_if_not_exist
def get_zhunbanlv0_data(p):
    _d = pd.read_csv(p, index_col=0)
    col = [_ for _ in _d.columns if _.startswith('本期')][0]
    return {
        'zhunbanlv_zonghe': _d.loc['综合准班率指数(%)', col],
        'zhunbanlv_daoligang': _d.loc['到离港服务准班率指数(%)', col],
        'zhunbanlv_shoufahuo': _d.loc['收发货服务准班率指数(%)', col],
    }
@ret_none_if_not_exist
def get_taiwanyunjia_data(p):
    _d = pd.read_csv(p, index_col=0)
    col = [_ for _ in _d.columns if _.startswith('本期')][0]
    return {
        'taiwan_zonghe': _d.loc['综合指数', col],
        'taiwan_chukou': _d.loc['出口成分指数', col],
    }
@ret_none_if_not_exist
def get_zhunbanlv1_data(p2):
    words = [__.strip().replace(',', '') for _ in open(p2, encoding='utf8').read().split() for __ in _.split(',')]

    def _isnumeric(s):
        try:
            float(s)
        except:
            return False
        return True

    is_num = [_isnumeric(_) for _ in words]
    record = dict()
    for i in range(len(words)-4):
        if is_num[i+1] and is_num[i+2] and is_num[i+3] and is_num[i+4]:
            record[words[i]] = [
                float(words[i+1]), float(words[i+2]),
                float(words[i+3]), float(words[i+4]),
            ]
    def _get(s):
        if s in record:
            return record[s][0]
        else:
            print(s)
            return np.nan
    return {
        'zhundianlv_eu': _get('亚洲-欧洲'),
        'zhundianlv_dizhonghai': _get('亚洲-地中海'),
        'zhundianlv_westus': _get('亚洲-美西'),
        'zhundianlv_eastus': _get('亚洲-美东'),
    }
@ret_none_if_not_exist
def get_jinkougansanhuo_data(p):
    _d = pd.read_csv(p, index_col=0)
    col = [_ for _ in _d.columns if _.startswith('指数')][0]
    def _find(_s):
        for idx in _d.index:
            if idx.find(_s)>=0:
                return _d.loc[idx, col]
        else:
            print(_s)
            return np.nan
    return {
        'jinkougansanhuo': _d.loc['综合指数', col],
        'jinkougansanhuo_tiekuangshi': _find('铁矿石'),
        'jinkougansanhuo_west_aus': _find('西澳大利亚'),
        'jinkougansanhuo_basitubalang': _find('巴西图巴朗'),
        'jinkougansanhuo_nanfei': _find('南非萨尔达尼亚'),
        'jinkougansanhuo_jialaerdun': _find('澳大利亚杰拉尔顿'),
        'jinkougansanhuo_meitan': _find('煤炭运价指数'),
        'jinkougansanhuo_niusikaer': _find('澳大利亚纽卡斯尔'),
        'jinkougansanhuo_haiboyinte': _find('澳大利亚海波因特'),
        'jinkougansanhuo_samalinda': _find('印尼萨马林达'),
        'jinkougansanhuo_tabaniao_guangzhou': _find('印尼塔巴尼奥-中国广州'),
        'jinkougansanhuo_tabaniao_nantong': _find('印尼塔巴尼奥—中国南通'),
        'jinkougansanhuo_dadou': _find('大豆'),
        'jinkougansanhuo_sangtuosi': _find('巴西桑托斯'),
        'jinkougansanhuo_takema': _find('美西塔科马'),
        'jinkougansanhuo_mixixibi': _find('美湾密西西比河'),
        'jinkougansanhuo_nie': _find('镍矿运价指数'),
        'jinkougansanhuo_suligao': _find('菲律宾苏里高'),
    }


@ret_none_if_not_exist
def get_sanhuozujin_data(p):
    _d = pd.read_csv(p, index_col=0)
    col = [_ for _ in _d.columns if _.find('日租金')>=0][0]
    return {
        'sanhuozujin_zonghe': float(_d.loc['综合指数', '单位']),
        'sanhuozujin_banama': _d[col].iloc[1],
        'sanhuozujin_dalingbian1': _d[col].iloc[2],
        'sanhuozujin_dalingbian2': _d[col].iloc[3],
        'sanhuozujin_lingbian1': _d[col].iloc[4],
        'sanhuozujin_lingbian2': _d[col].iloc[5],
        'sanhuozujin_lingbian3': _d[col].iloc[6],
    }
@ret_none_if_not_exist
def get_jinkoujizhuangxiang_data(p2):
    words = [__.strip().replace(',', '') for _ in open(p2, encoding='utf8').read().split() for __ in _.split(',')]
    words = [_ for _ in words if _]

    def _isnumeric(s):
        try:
            float(s)
        except:
            return False
        return True

    is_num = [_isnumeric(_) for _ in words]
    record = dict()
    for i in range(len(words)-3):
        if is_num[i+1] and is_num[i+2] and is_num[i+3]:
            record[words[i]] = [float(words[i+1]),
                                float(words[i+2]),
                                float(words[i+3])]
    def _get(s):
        if s in record:
            return record[s][1]
        else:
            print(s)
            return np.nan
    return {
        'jinkou_zonghe': _get('综合指数'),
        'jinkou_ouzhou': _get('欧洲航线'),
        'jinkou_dizhonghai': _get('地中海航线'),
        'jinkou_meixi': _get('美西航线'),
        'jinkou_meidong': _get('美东航线'),
        'jinkou_nanfei': _get('南非航线'),
        'jinkou_nanmei': _get('南美航线'),
    }
@ret_none_if_not_exist
def get_jinkouhaiyun_data(p2):
    words = [__.strip().replace(',', '') for _ in open(p2, encoding='utf8').read().split() for __ in _.split(',')]

    def _isnumeric(s):
        try:
            float(s)
        except:
            return False
        return True

    is_num = [_isnumeric(_) for _ in words]
    record = dict()
    for i in range(len(words)-2):
        if is_num[i+1] and is_num[i+2]:
            record[words[i]] = [float(words[i+1]),
                                float(words[i+2]),]
    def _get(s):
        for ss, v in record.items():
            if ss.find(s)>=0:
                return v[0]
        else:
            print(s)
            return np.nan
    return {
        'jinkou_haiyun_eu': _get('欧洲'),
        'jinkou_haiyun_easteu': _get('东南欧'),
        'jinkou_haiyun_westeu': _get('西北欧'),
        'jinkou_haiyun_eastasia': _get('东南亚'),
        'jinkou_haiyun_southasia': _get('（三）南亚'),
        'jinkou_haiyun_westasia': _get('西亚'),
        'jinkou_haiyun_southam': _get('南美洲'),
        'jinkou_haiyun_westus': _get('美国（西岸）'),
        'jinkou_haiyun_eastus': _get('美国（东岸）'),
        'jinkou_haiyun_aus': _get('大洋洲'),
        'jinkou_haiyun_ydyl': _get('一带一路'),

    }

def get_data_one_folder(p):
    d_list = list()
    d_list.append(get_yidaiyilu_data(p+'“一带一路” 贸易额指数.0.csv'))
    d_list.append(get_yidaiyilu_haiyun_data(p+'“一带一路”集装箱海运量指数.0.csv'))
    d_list.append(get_sichouzhilu_data(p+'“海上丝绸之路”运价指数.0.csv'))
    d_list.append(get_sh_chukoujiesuan_data(p+'上海出口集装箱结算运价指数.0.csv'))
    d_list.append(get_sh_chukouyunjia_data(p+'上海出口集装箱运价指数.0.csv'))
    d_list.append(get_dongnanyayunjia_data(p+'东南亚集装箱运价指数.0.csv'))
    d_list.append(get_zg_chukouyunjia_data(p+'中国出口集装箱运价指数.0.csv'))
    d_list.append(get_chengpinyou_data(p+'中国沿海成品油运价指数.0.csv'))
    d_list.append(get_sanhuozujin_data(p+'中国沿海散货船舶日租金指数.0.csv'))
    d_list.append(get_sanhuoyunjia_data(p+'中国沿海散货运价指数.0.csv'))
    d_list.append(get_meitanyunjia_data(p+'中国沿海煤炭运价指数.0.csv'))
    d_list.append(get_liangshiyunjia_data(p+'中国沿海粮食运价指数.0.csv'))
    d_list.append(get_jinshukuangshi_data(p+'中国沿海金属矿石运价指数.0.csv'))
    d_list.append(get_jinkouyuanyou_data(p+'中国进口原油运价指数.0.csv'))
    d_list.append(get_jinkougansanhuo_data(p+'中国进口干散货运价指数.0.csv'))
    d_list.append(get_jinkoujizhuangxiang_data(p+'中国进口集装箱运价指数.0.csv'))
    # d_list.append(get_haiyuanxinchou0_data(p+'中国（上海）海员薪酬指数.0.csv'))
    # d_list.append(get_haiyuanxinchou1_data(p+'中国（上海）海员薪酬指数.1.csv'))
    d_list.append(get_jinkouhaiyun_data(p+'中国（上海）进口贸易海运指数.0.csv'))
    d_list.append(get_zhunbanlv0_data(p+'全球集装箱班轮准班率指数.0.csv'))
    d_list.append(get_zhunbanlv1_data(p+'全球集装箱班轮准班率指数.1.csv'))
    d_list.append(get_taiwanyunjia_data(p+'台湾海峡两岸间集装箱运价指数.0.csv'))
    # d_list.append(get_yuandonggansanhuo_data(p+'远东干散货指数.0.csv'))
    k = [__ for _ in d_list if _ is not None for __ in _.keys()]
    assert len(k) == len(set(k))
    d = dict()
    for dd in d_list:
        if dd:
            d.update(dd)
    return d


def main():
    dates = [_ for _ in os.listdir(p_index) if os.path.isdir(p_index+_)]
    paths = [(_t, p_index+_d+'/'+_t+'/') for _d in dates for _t in os.listdir(p_index+_d)]
    data_list = []
    for t, p in tqdm.tqdm(paths):
        data = get_data_one_folder(p)
        data['time'] = t
        data_list.append(data)
    data_df = pd.DataFrame(data_list)
    data_df.to_csv(p_out+'data_index.csv', index=False)
    data_df.to_pickle(p_out+'data_index.pkl')
    this_log.info('save index data {} to {}'.format(data_df.shape, p_out+'data_index.pkl'))

    cols = [_ for _ in data_df.columns if _ != 'time']
    data_df['year'] = data_df['time'].str.slice(0, 4)
    data_df['month'] = data_df['time'].str.slice(4, 6)
    data_df['day'] = data_df['time'].str.slice(6, 8)
    data_df['hour'] = data_df['time'].str.slice(8, 10)
    data_df['weekday'] = data_df['time'].apply(lambda x: datetime.datetime.strptime(x, '%Y%m%d%H%M%S').strftime('%A'))
    is_change = (data_df[cols].astype(float).diff()!=0)
    is_change = (data_df[['chukoujiesuan_eu']].astype(float).diff()!=0).any(axis=1)
    data_df['is_change'] = is_change
    data_df[is_change].groupby(['weekday', 'hour'])['year'].count()

    print(data_df.dtypes.to_string())


if __name__ == "__main__":
    main()
