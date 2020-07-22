import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import  MinMaxScaler
import sklearn.metrics as metrics
import matplotlib as mpl
import math
from utils.plot_res import plot_results
from utils.eval_res import eva_regress
 
import warnings
warnings.filterwarnings("ignore")
 
 
'''  多输入并行序列预测问题：
        'Lane 1 Flow (Veh/5 Minutes)' 和 'line2'两个车道下的过车数量
        即用两个车道上过去12个时间切片的数据，预测当前时刻，两个车道的过车数量
        
        输入数据：同一时刻，两个车道的过车数量，
        输出结果：同时预测当前时刻两个车道的过车数量
    
'''
 
attr = 'Lane 1 Flow (Veh/5 Minutes)'
 
training_set = pd.read_csv('data/train_add.csv')
training_set = training_set.loc[:,[attr,'line2']].values
 
test_set = pd.read_csv('data/test_add.csv')
test_set = test_set.loc[:,[attr,'line2']].values
 
 
sc = MinMaxScaler()
training_data = sc.fit_transform(training_set)
 
 
# 数据切分：用过去12个数据预测当前数据
def sliding_windows(seq,n_steps):
    x,y=[],[]
    for i in range(len(seq)):
        end_ix = i + n_steps
        if end_ix>len(seq)-1:
            break
        seq_x = seq[i:end_ix,:]
        seq_y = seq[end_ix,:]
        x.append(seq_x)
        y.append(seq_y)
    return np.array(x),np.array(y)
 
n_steps = 12  # 步长12
x,y = sliding_windows(training_data,n_steps)
 
# 将训练集进一部分为训练集和验证集
train_size = int(len(x)*0.67)
# test_size = len(x) - train_size
 
dataX = Variable(torch.Tensor(x))
dataY = Variable(torch.Tensor(y))
 
# train.csv中用来训练的数据部分
trainX = Variable(torch.Tensor(x[0:train_size]))
trainY = Variable(torch.Tensor(y[0:train_size]))
 
 
 
# train.csv中用来验证的数据部分
valdX = Variable(torch.Tensor(x[train_size:]))
valdY = Variable(torch.Tensor(y[train_size:]))
 
# 构建模型
class LSTM_Predict(nn.Module):
    def __init__(self,num_classes,input_size,hidden_size,num_layers):
        super(LSTM_Predict,self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.n_steps = n_steps
 
        self.lstm = nn.LSTM(input_size=input_size,hidden_size=hidden_size,
                            num_layers=num_layers,batch_first=True)
 
        self.fc = nn.Linear(hidden_size,num_classes)
 
    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers,x.size(0),
                                   self.hidden_size))
        c_0 = Variable(torch.zeros(self.num_layers,x.size(0),
                                   self.hidden_size))
 
        ula,(h_out,_) = self.lstm(x,(h_0,c_0))
        h_out = h_out.view(-1,self.hidden_size)
 
        out = self.fc(h_out)
        return out
 
 
num_epochs = 1000
learning_rate = 0.01
 
input_size = 2
hidden_size = 64
num_layers = 1
num_classes = 2
 
lstm = LSTM_Predict(num_classes,input_size,hidden_size,num_layers)
 
# MSE 误差
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(lstm.parameters(),lr=learning_rate)
 
loss_his =[]
# 轮询输出结果
for epoch in range(num_epochs):
    # 相当于默认调用forward前向传播算法
    outputs = lstm(trainX)
    optimizer.zero_grad()
 
    # 获得损失函数
    loss = criterion(outputs,trainY)
    loss_his.append(loss.item())
    # 反向传播求梯度
    loss.backward()
    # 更新所有参数
    optimizer.step()
    if epoch%100==0:
        print("Epoch: %d, Loss: %1.5f" % (epoch,loss.item()))
 
 
# plt.plot([i for i in range(num_epochs)], loss_his, '--', label='Predictions',alpha=0.5)
# plt.show()
 
 
# model prediction
lstm.eval()
train_predict = lstm(dataX)
 
 
dataY_predict = train_predict.data.numpy()
dataY_true = dataY.data.numpy()
# dataY_true = dataY_true.reshape(dataY_true.shape[0],dataY_true.shape[2])
 
 
dataY_predict = sc.inverse_transform(dataY_predict)
dataY_true = sc.inverse_transform(dataY_true)
 
print('ok')
 
 
# Lane 1的预测结果与真实结果对比分析，取前288个数据进行比较
plt.plot(dataY_predict[:288,0],c='r', linestyle='--')
plt.plot(dataY_true[:288,0],c='b', linestyle='--')
plt.suptitle('the comprison of lane 1')
plt.show()
 
# Line 2的预测结果与真实结果对比分析，取前288个数据进行比较
plt.plot(dataY_predict[:288,1],c='r', linestyle='--')
plt.plot(dataY_true[:288,1],c='b', linestyle='--')
plt.suptitle('the comprison of line 2')
plt.show()
 
 
 
 
  
'''
    测试集数据进行预测分析
'''
 
 
# 测试集归一化处理
test_data = sc.fit_transform(test_set)
 
# 测试集切分
test_x,test_y = sliding_windows(test_data,n_steps)
 
test_x = Variable(torch.Tensor(test_x))
test_y = Variable(torch.Tensor(test_y))
 
# 利用训练好的模型对测试集进行预测
lstm.eval()
test_predict = lstm(test_x)
 
# 将预测结果和真实数据结果转换为numpy的格式
test_predict = test_predict.data.numpy()
testY_true = test_y.data.numpy()
 
# 将预测结果和真实数据转换为原始数据范围
test_predict1 = sc.inverse_transform(test_predict)
testY_true1 = sc.inverse_transform(testY_true)
 
 
# 车道lane1的预测结果与真实结果对比
test_predict_lane1 = test_predict1[:,0]
testY_true_lane1 = testY_true1[:,0]
plt.plot(test_predict_lane1[:288],c='r', linestyle='--')
plt.plot(testY_true_lane1[:288],c='b', linestyle='--')
plt.suptitle('testing Traffic Flow Prediction of lane 1')
plt.show()
 
 
# 车道line2的预测结果与真实结果对比
test_predict_lane2 = test_predict1[:,1]
testY_true_lane2 = testY_true1[:,1]
plt.plot(test_predict_lane2[:288],c='r', linestyle='--')
plt.plot(testY_true_lane2[:288],c='b', linestyle='--')
plt.suptitle('testing Traffic Flow Prediction of line 2')
plt.show()
 
 
# 将两个车道的预测结果flatten拉平，综合查看模型各个评价指标的结果
test_predict_total = test_predict1.reshape(1,-1)[0]
testY_true_total = testY_true1.reshape(1,-1)[0]
# 回归模型的各个指标评价：MSE,MAE,r2,MAPE
eva_regress(testY_true_total, test_predict_total)
 
 
 
# 车道lane1的预测结果与真实结果对比
y_preds = []
test_predict_lane1 = test_predict1[:,0].reshape(1,-1)[0]   #转换为行数据
testY_true_lane1 = testY_true1[:,0].reshape(1,-1)[0]       #转换为行数据
 
y_preds.append(test_predict_lane1[0:288])
 
plot_results(testY_true_lane1[:288], y_preds, names='LSTM')
 
# 车道line2的预测结果与真实结果对比
y_preds = []
test_predict_lane2 = test_predict1[:,1].reshape(1,-1)[0]   #转换为行数据
testY_true_lane2 = testY_true1[:,1].reshape(1,-1)[0]       #转换为行数据
 
y_preds.append(test_predict_lane2[0:288])
 
plot_results(testY_true_lane2[:288], y_preds, names='LSTM')

 
 
