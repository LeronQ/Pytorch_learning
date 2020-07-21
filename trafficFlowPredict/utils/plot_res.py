import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import warnings
 
 
def plot_results(y_true, y_preds, names):
    """
    画出结果图
    """
    d = '2016-3-4 00:00'
    x = pd.date_range(d, periods=288, freq='5min')
 
    fig = plt.figure()
    ax = fig.add_subplot(111)
 
    ax.plot(x, y_true, label='True Data')
    for name, y_pred in zip(names, y_preds):
        ax.plot(x, y_pred, label=name)
 
    plt.legend()
    plt.grid(True)
    plt.xlabel('Time of Day')
    plt.ylabel('Flow')
 
    date_format = mpl.dates.DateFormatter("%H:%M")
    ax.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()
 
    plt.show()
