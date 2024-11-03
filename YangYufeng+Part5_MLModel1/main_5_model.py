#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import datetime
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
import lightgbm as lgb
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Lasso
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# from my_logger import this_log


p_in = r'E:\Document\MyStrategy\DEDA_fdu24fall_finalproj_group1\final\output' + '\\'
p_out = r'E:\Document\MyStrategy\DEDA_fdu24fall_finalproj_group1\final\output' + '\\'
p_factor = p_in+'factor.pkl'
p_target = p_in+'target.pkl'


def train_lgbm(train_x, train_y, valid_x, valid_y, test_x, test_y):
    train_data = lgb.Dataset(train_x, label=train_y)
    valid_data = lgb.Dataset(valid_x, label=valid_y)

    params = {
        'objective': 'regression',
        'metric': 'mse',
        'boosting_type': 'gbdt',
        'learning_rate': 0.01,
        'num_leaves': 8,
        'max_depth': 3,
        'verbosity': 100,
        'early_stopping_rounds': 200,
    }
    evals = [train_data, valid_data]

    def pos_corr(y_pred, y_true):
        # Convert to numpy arrays if not already
        y_true = y_true.get_label()
        correlation = np.corrcoef(y_pred, y_true)[0, 1]
        return 'pos_corr', correlation, True  # Return negative correlation

    res = []
    def callback(x):
        res.append(x)
    model = lgb.train(
        params, train_data,
        feval=pos_corr,
        num_boost_round=10000, valid_sets=evals,
        callbacks=[callback]
    )
    y_pred = model.predict(test_x, num_iteration=model.best_iteration)
    mse = mean_squared_error(test_y, y_pred)
    print(f"Mean Squared Error: {mse}")
    # plt.figure(figsize=(10, 6))
    # lgb.plot_metric(model, metric='mse', title='Learning Curve', xlabel='Iterations', ylabel='MSE')
    # plt.show()
    learning_curve = pd.DataFrame([{
        'train': _.model.best_score['training']['pos_corr'],
        'valid': _.model.best_score['valid_1']['pos_corr']
    } for _ in res])
    learning_curve.plot()
    plt.show()
    return model, y_pred


def train_model(train_x, train_y, valid_x, valid_y, model_name):
    if model_name == 'linear_regression':
        model = LinearRegression()
    elif model_name == 'random_forest':
        model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    elif model_name == 'lgbm':
        model = LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    elif model_name == 'neural_network':
        model = MLPRegressor(hidden_layer_sizes=(16, 8, 4),  # Two hidden layers
                           activation='relu',  # Activation function
                           solver='adam',  # Optimizer
                           learning_rate_init=0.001,  # Initial learning rate
                           batch_size=32,  # Batch size
                           max_iter=500,  # Maximum epochs
                           early_stopping=True,  # Enable early stopping
                           validation_fraction=0.1,  # Fraction of training data for validation
                           n_iter_no_change=20,  # Early stopping patience
                           random_state=42)
    elif model_name == 'lasso':
        model = Lasso(alpha=0.1)  # You can adjust alpha for regularization strength
    else:
        raise ValueError(f"Model {model_name} not supported")

    # Train model
    model.fit(train_x, train_y)
    return model


# Prediction function
def predict_model(model, test_x):
    return model.predict(test_x)


# Evaluation function
def evaluate_model(predictions, test_y, test_y_benchmark=None):
    mse = mean_squared_error(test_y, predictions)
    mae = mean_absolute_error(test_y, predictions)
    r2 = r2_score(test_y, predictions)
    df = pd.DataFrame({'pv': predictions, 'true': test_y})
    corr = df.corr().iloc[0, 1]
    df['pv_group'] = np.floor(df['pv'].rank(pct=True)*5).astype(int)
    group_mean = df.groupby('pv_group')['true'].mean()
    group_mean.index = ['group'+str(_) for _ in group_mean.index]
    group_mean.plot()
    plt.show()
    # benchmark_mse = mean_squared_error(test_y, test_y_benchmark)

    sign_accuracy = np.mean(np.sign(predictions) == np.sign(test_y))

    # 2. Precision and Recall
    # Convert to binary classification (e.g., positive if > 0, negative otherwise)
    predicted_binary = (predictions > 0).astype(int)
    real_binary = (test_y > 0).astype(int)

    precision = precision_score(real_binary, predicted_binary)
    recall = recall_score(real_binary, predicted_binary)

    # 3. AUC
    auc = roc_auc_score(real_binary, predictions)

    eval_result = {
        "mse": mse,
        "mae": mae,
        "r2": r2,
        # "benchmark_mse": benchmark_mse
        'corr': corr,
        'accuracy': sign_accuracy,
        'precision': precision,
        'recall': recall,
        'auc': auc,
    }
    # eval_result.update(group_mean.to_dict())
    return eval_result



def winsorize_and_standardize_fit(df):
    parameters = {}
    for col in df.columns:
        median = df[col].median()
        bottom_1 = df[col].quantile(0.01)
        top_99 = df[col].quantile(0.99)
        lower_bound = median - 2 * (median - bottom_1)
        upper_bound = median + 2 * (top_99 - median)
        clipped_col = df[col].clip(lower=lower_bound, upper=upper_bound)
        median2 = clipped_col.median()
        clipped_col = clipped_col.fillna(median2)
        mean = clipped_col.mean()
        std = clipped_col.std()
        parameters[col] = {
            'median': median,
            'median2': median2,
            'bottom_1': bottom_1,
            'top_99': top_99,
            'mean': mean,
            'std': std,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }
    return parameters


def winsorize_and_standardize_fit_apply(df, parameters):
    transformed_df = df.copy()
    for col in df.columns:
        param = parameters[col]
        clipped_col = df[col].clip(lower=param['lower_bound'], upper=param['upper_bound'])
        clipped_col = clipped_col.fillna(param['median2'])
        # standardized_col = (clipped_col - param['mean']) / param['std']
        # transformed_df[col] = standardized_col
        transformed_df[col] = clipped_col
    return transformed_df


def main_daily_model():
    factor = pd.read_pickle(p_factor)
    target_df = pd.read_pickle(p_target)

    t = list(factor.index)
    date = [_.strftime('%Y%m%d') for _ in t]
    factor['date'] = date
    factor_daily = factor.drop_duplicates('date', keep='last')
    factor_daily = factor_daily.set_index('date')

    target_daily = pd.DataFrame(index=factor_daily.index)
    target_daily['preclose_to_open'] = factor_daily['open'].shift(-1)/factor_daily['mid']-1
    target_daily['open_to_close'] = (factor_daily['mid']/factor_daily['open']-1).shift(-1)
    target_daily['close_to_close'] = (factor_daily['mid'].shift(-1)/factor_daily['mid']-1)
    target_daily['ret_index'] = factor_daily['eu_chukou'].shift(-1)/factor_daily['eu_chukou']-1
    target_daily = target_daily.fillna(0)

    factor_daily.to_pickle(p_out+'factor_daily.pkl')
    target_daily.to_pickle(p_out+'target_daily.pkl')

    target_name = 'preclose_to_open'
    target_name = 'close_to_close'
    target_name = 'ret_index'

    # target_df = target_df.fillna(0)
    target = target_daily[target_name]
    target_benchmark = target_daily[target_name]
    # corr = factor.corrwith(target)
    # corr.abs().sort_values().dropna()
    # corr.abs().describe()
    # target.describe()

    train_pct, valid_pct = 0.6, 0.2
    idx_train = factor_daily.index[:int(len(factor_daily)*train_pct)]
    idx_valid = factor_daily.index[int(len(factor_daily)*train_pct):int(len(factor_daily)*(train_pct+valid_pct))]
    idx_test = factor_daily.index[int(len(factor_daily)*(train_pct+valid_pct)):]

    train_x0, valid_x0, test_x0 = factor_daily.loc[idx_train], factor_daily.loc[idx_valid], factor_daily.loc[idx_test]
    train_y, valid_y, test_y = target.loc[idx_train], target.loc[idx_valid], target.loc[idx_test]
    test_y_benchmark = target_benchmark.loc[idx_test]

    winso_para = winsorize_and_standardize_fit(train_x0)
    has_na = pd.DataFrame(winso_para).isna().any()
    factor_col = has_na[~has_na].index
    train_x1 = winsorize_and_standardize_fit_apply(train_x0[factor_col], winso_para)
    corr = train_x1.corrwith(target)
    factor_col_select = corr[corr.abs() >= 0.20].index

    train_x = winsorize_and_standardize_fit_apply(train_x0[factor_col_select], winso_para)
    valid_x = winsorize_and_standardize_fit_apply(valid_x0[factor_col_select], winso_para)
    test_x = winsorize_and_standardize_fit_apply(test_x0[factor_col_select], winso_para)

    pv_dict = dict()
    eval_dict = dict()
    model_dict = dict()
    model_name_list = [
        'linear_regression',
        'random_forest',
        'lasso',
        'lgbm',
        # 'neural_network'
    ]
    for model_name in model_name_list:
        print('{} begin'.format(model_name))
        model = train_model(train_x, train_y, valid_x, valid_y, model_name)
        # pv = pd.Series(predict_model(model, train_x), train_x.index)
        # eval_res_ = evaluate_model(pv, train_y, train_y)
        pv = pd.Series(predict_model(model, test_x), test_x.index)
        eval_res_ = evaluate_model(pv, test_y, test_y_benchmark)
        pv_dict[model_name] = pv
        model_dict[model_name] = model
        eval_dict[model_name] = eval_res_

    model_lgbm, pv_lgbm = train_lgbm(train_x, train_y, valid_x, valid_y, test_x, test_y)
    eval_dict['lgbm_fine_tuned'] = evaluate_model(pv_lgbm, test_y, test_y_benchmark)
    eval_df = pd.DataFrame(eval_dict).T
    print(eval_df.to_string())
    # eval_df.to_csv(p_out+'evaluation_daily.csv')

    fi = pd.Series(model_lgbm.feature_importance(), index=train_x.columns).sort_values(ascending=True)
    fi = fi[fi > 0]
    fi = fi.rename({
        'jinkou_nanmei_pct_1w': 'south_af_imp_ret',
        'chukoujiesuan_eu_pct_1w': 'euro_ex_settle_ret',
        'eu_chukou_pct_1w': 'euro_ex_close_ret',
        'oib5_10min': 'fut_order_imba',
        'sentiment_redsea_tot_num_d1': 'redsea_sentiment',
        'jinkougansanhuo_jialaerdun_pct_1w': 'dry_bulk_imp_ret',
        'sanhuo_yunjia_pct_1w': 'dry_bulk_price_ret',
        'jinkou_haiyun_eu': 'euro_import_price',
        'sanhuo_yunjia': 'dry_bulk_price',
        'south_africa_chukou_pct_1w': 'south_af_ex_ret',
        'mediterranean_chukou_pct_1w': 'mediterr_ex_ret'
    })
    plt.rcParams['font.size'] = 20  # Set the default font size
    plt.figure(figsize=(22,15))
    plt.barh(fi.index, fi.values)
    plt.title('feature importance')
    # plt.grid()
    plt.savefig(p_out+'feature_importance.png')
    plt.show()


if __name__ == "__main__":
    main_daily_model()