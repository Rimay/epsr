
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import xgboost as xgb
from vis import eval
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from IPython import embed




# 定义ARIMA模型, 包括模型训练, 模型评估, 模型可视化
def arima_model(dfd):
    train, test = dfd[:'2009'], dfd['2010':]
    # 模型训练
    model = ARIMA(train, order=(1, 1, 1))
    model_fit = model.fit()
    # 模型评估
    pred = model_fit.forecast(steps=len(test))
    pred = pd.Series(pred.values, index=test.index)
    eval(test, pred)
    
    plt.plot(pred, c='red', label='ARIMA prediction')
    plt.plot(test, c='green', label='true value')
    plt.legend()
    plt.show()
    return model_fit

def xgb_model(dfd):
    dfd = dfd.to_frame()
    for i in range(1, 8):
        dfd[f'pre_day_{i}'] = dfd['Global_active_power'].shift(i)
    dfd.dropna(inplace=True)
    train, test = dfd[:'2009'], dfd['2010':]

    x_train, y_train, x_test, y_test = train.drop(columns=['Global_active_power']), train['Global_active_power'],\
                                         test.drop(columns=['Global_active_power']), test['Global_active_power']

    # 模型训练
    model = xgb.XGBRegressor()
    model.fit(x_train, y_train)
    # 模型预测
    pred = model.predict(x_test)
    pred = pd.Series(pred, index=y_test.index)

    eval(y_test, pred)

    plt.figure(figsize=(12, 6))
    plt.plot(pred, c='red', label='XGBRegressor prediction')
    plt.plot(y_test, c='green', label='true value')
    plt.legend()
    plt.xticks(rotation=45, fontsize=8)
    plt.tight_layout()
    plt.show()
    return model


