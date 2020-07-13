
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import  MinMaxScaler

'''
    单变量预测模型:
        包括LSTM时序预测层和Linear回归输出层,可以根据自己的情况增加模型结构
'''

training_set = pd.read_csv('airline-passengers.csv')
#training_set = pd.read_csv('shampoo.csv')
training_set = training_set.iloc[:,1:2].values


def sliding_windows(seq,n_steps):
    x,y = [],[]
    for i in range(len(seq)):
        end_ix = i + n_steps
        if end_ix > len(seq)-1:
            break
        seq_x = seq[i:end_ix]
        seq_y = seq[end_ix]
        x.append(seq_x)
        y.append(seq_y)
    return np.array(x),np.array(y)

# 数据归一化
scaler = MinMaxScaler()
training_data = scaler.fit_transform(training_set)

# 将数据转换为训练数据格式
n_steps = 4
x,y = sliding_windows(training_data,n_steps)

# 切分数据位训练集和测试集
train_size = int(len(x)*0.67)
test_size = len(x) - train_size

dataX = Variable(torch.Tensor(x))
dataY = Variable(torch.Tensor(y))

trainX = Variable(torch.Tensor(x[0:train_size]))
trainY = Variable(torch.Tensor(y[0:train_size]))

testX = Variable(torch.Tensor(x[train_size:]))
testY = Variable(torch.Tensor(y[train_size:]))

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


num_epochs = 2000
learning_rate = 0.01

input_size = 1
hidden_size = 2
num_layers = 1
num_classes = 1

lstm = LSTM_Predict(num_classes,input_size,hidden_size,num_layers)

# MSE 误差
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(lstm.parameters(),lr=learning_rate)


# 轮询输出结果
for epoch in range(num_epochs):
    # 相当于默认调用forward前向传播算法
    outputs = lstm(trainX)
    optimizer.zero_grad()

    # 获得损失函数
    loss = criterion(outputs,trainY)
    # 反向传播求梯度
    loss.backward()
    # 更新所有参数
    optimizer.step()
    if epoch%100==0:
        print("Epoch: %d, Loss: %1.5f" % (epoch,loss.item()))


#Testing for passengers Dataset

lstm.eval()
train_predict = lstm(dataX)


data_predict = train_predict.data.numpy()
dataY_plot = dataY.data.numpy()

data_predict = sc.inverse_transform(data_predict)
dataY_plot = sc.inverse_transform(dataY_plot)

plt.axvline(x=train_size, c='r', linestyle='--')

plt.plot(dataY_plot)
plt.plot(data_predict)
plt.suptitle('Time-Series Prediction')
plt.show()

'''
    # gradient descent
    weights = [0] * n
    alpha = 0.0001
    max_Iter = 50000
    for i in range(max_Iter):
        loss = 0
        d_weights = [0] * n
        for k in range(m):
            h = dot(input[k], weights)
            d_weights = [d_weights[j] + (label[k] - h) * input[k][j] for j in range(n)] 
            loss += (label[k] - h) * (label[k] - h) / 2
        d_weights = [d_weights[k]/m for k in range(n)]
        weights = [weights[k] + alpha * d_weights[k] for k in range(n)]
        if i%10000 == 0:
            print "Iteration %d loss: %f"%(i, loss/m)
            print weights

发现它们实际上是一一对应的：

optimizer.zero_grad()对应d_weights = [0] * n

即将梯度初始化为零（因为一个batch的loss关于weight的导数是所有sample的loss关于weight的导数的累加和）

outputs = net(inputs)对应h = dot(input[k], weights)

即前向传播求出预测的值

loss = criterion(outputs, labels)对应loss += (label[k] - h) * (label[k] - h) / 2

这一步很明显，就是求loss（其实我觉得这一步不用也可以，反向传播时用不到loss值，只是为了让我们知道当前的loss是多少）
loss.backward()对应d_weights = [d_weights[j] + (label[k] - h) * input[k][j] for j in range(n)]

即反向传播求梯度
optimizer.step()对应weights = [weights[k] + alpha * d_weights[k] for k in range(n)]

即更新所有参数

'''