import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import warnings
 
 
'''
    回归模型的各个指标评价：MSE,MAE,r2,MAPE
'''
def MAPE(y_true, y_pred):
    """Mean Absolute Percentage Error
    均方差误差百分比结果
    """
 
    y = [x for x in y_true if x > 0]
    y_pred = [y_pred[i] for i in range(len(y_true)) if y_true[i] > 0]
 
    num = len(y_pred)
    sums = 0
 
    for i in range(num):
        tmp = abs(y[i] - y_pred[i]) / y[i]
        sums += tmp
 
    mape = sums * (100 / num)
 
    return mape
 
 
def eva_regress(y_true, y_pred):
    """
    :param y_true:
    :param y_pred:
    :return:  回归模型评价结果
    """
 
    mape = MAPE(y_true, y_pred)
    vs = metrics.explained_variance_score(y_true, y_pred)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)
    print('explained_variance_score:%f' % vs)
    print('mape:%f%%' % mape)
    print('mae:%f' % mae)
    print('mse:%f' % mse)
    print('rmse:%f' % math.sqrt(mse))
    print('r2:%f' % r2)
 
