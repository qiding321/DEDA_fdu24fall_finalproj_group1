import numpy as np
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd

# 数据加载
df = pd.read_csv("final_data.csv")
df = df.dropna()

# -1期变量的值作为特征变量预测
df.loc[:, 'S_1'] = df['close_price'].shift(1).rolling(window=1).mean()
df = df.dropna()

# 特征变量和目标变量
X = df[['S_1', 'return', 'EUROPE']]
y = df['close_price']

# 分训练和测试集，比例为82%
t = 0.8
t = int(t * len(df))

# 训练集
X_train = X[:t]
y_train = y[:t]

# 测试集
X_test = X[t:]
y_test = y[t:]

# 模型训练和预测
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(),
    "Support Vector Machine": SVR(kernel='linear'),
    "KNN": KNeighborsRegressor(),
    "Bayesian Ridge": BayesianRidge()
}

results = []

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # 计算评价指标
    r2 = r2_score(y_test, y_test_pred)
    mse = mean_squared_error(y_test, y_test_pred)
    mae = mean_absolute_error(y_test, y_test_pred)

    results.append({
        "Model": model_name,
        "R²": r2,
        "MSE": mse,
        "MAE": mae
    })

# 汇总结果到 DataFrame
results_df = pd.DataFrame(results)

# 打印结果
print(results_df)

# 绘图
train_predictions = pd.DataFrame({
    'LR': models["Linear Regression"].predict(X_train),
    'RFR': models["Random Forest"].predict(X_train),
    'SVR': models["Support Vector Machine"].predict(X_train),
    'KNN': models["KNN"].predict(X_train),
    'BR': models["Bayesian Ridge"].predict(X_train)
}, index=y_train.index)

test_predictions = pd.DataFrame({
    'LR': models["Linear Regression"].predict(X_test),
    'RFR': models["Random Forest"].predict(X_test),
    'SVR': models["Support Vector Machine"].predict(X_test),
    'KNN': models["KNN"].predict(X_test),
    'BR': models["Bayesian Ridge"].predict(X_test)
}, index=y_test.index)

predictions = pd.concat([train_predictions, test_predictions])

# 绘图
plt.figure(figsize=(10, 8))
plt.title("Predictions vs Actual Prices")
plt.plot(y_test, label="Actual Price")
plt.plot(test_predictions['LR'], label="Linear Regression")
plt.plot(test_predictions['RFR'], label="Random Forest")
plt.plot(test_predictions['SVR'], label="Support Vector Machine")
plt.plot(test_predictions['KNN'], label="KNN")
plt.plot(test_predictions['BR'], label="Bayesian Ridge")
plt.legend()
plt.savefig('predictions_vs_actual_price.png', dpi=300)
plt.show()
