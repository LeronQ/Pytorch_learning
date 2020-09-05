# @Time    : 2020/7/30 
# @Author  : LeronQ
# @github  : https://github.com/LeronQ


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
import torch.nn.init as init

from utils.plot_res import plot_results
from utils.eval_res import eva_regress
import torch.utils.data as Data


import warnings
warnings.filterwarnings("ignore")
'''
    单变量预测模型:
        包括LSTM时序预测层和Linear回归输出层,可以根据自己的情况增加模型结构
'''

# 加载训练和测试数据集
training_set = pd.read_csv('data/train.csv')
training_set = training_set.iloc[:,1:2].values

test_set = pd.read_csv('data/test.csv')
test_set = test_set.iloc[:,1:2].values

# 训练数据归一化
sc = MinMaxScaler()
training_data = sc.fit_transform(training_set)

# 数据切分：用过去12个数据预测当前数据
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
        # h_0 = Variable(torch.zeros(self.num_layers,x.size(0),
        #                            self.hidden_size))
        # c_0 = Variable(torch.zeros(self.num_layers,x.size(0),
        #                            self.hidden_size))

        h_0 = nn.Parameter(torch.Tensor(self.num_layers,x.size(0),self.hidden_size)).to(device)
        c_0 = nn.Parameter(torch.Tensor(self.num_layers,x.size(0),self.hidden_size)).to(device)
        init.xavier_normal_(h_0)
        init.xavier_normal_(c_0)

        ula,(h_out,_) = self.lstm(x,(h_0,c_0))
        h_out = h_out.view(-1,self.hidden_size)

        out = self.fc(h_out)
        return out


num_epochs = 20
learning_rate = 0.01

input_size = 1
hidden_size = 64
num_layers = 1
num_classes = 1

lstm = LSTM_Predict(num_classes,input_size,hidden_size,num_layers)

# MSE 误差
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(lstm.parameters(),lr=learning_rate)

loss_his =[]

MINIBATCH_SIZE = 32
torch_dataset = Data.TensorDataset(trainX, trainY)
# put the dataset into DataLoader
'''
DataLoader 是 Torch 用来包装数据的工具，
如将 numpy array 等数据形式转成 Tensor，然后放进该包装器中。使用 DataLoader 可以有效迭代数据
'''
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=MINIBATCH_SIZE,
    shuffle=False
)
# 轮询输出结果

for epoch in range(num_epochs):
    for step, (batch_x, batch_y) in enumerate(loader):
        outputs = lstm(batch_x)
        optimizer.zero_grad()

        # 获得损失函数
        loss = criterion(outputs, batch_y)
        loss_his.append(loss.item())
        # 反向传播求梯度
        loss.backward()
        # 更新所有参数
        optimizer.step()
    if epoch % 100 == 0:
        print("Epoch: %d, Loss: %1.5f" % (epoch, loss.item()))


# plt.plot([i for i in range(num_epochs)], loss_his, '--', label='Predictions',alpha=0.5)
# plt.show()


# model prediction
lstm.eval()
train_predict = lstm(dataX)


dataY_predict = train_predict.data.numpy()
dataY_true = dataY.data.numpy()

dataY_predict = sc.inverse_transform(dataY_predict)
dataY_true = sc.inverse_transform(dataY_true)

# plt.axvline(x=train_size, c='r', linestyle='--')
plt.plot(dataY_predict[5201:],c='r', linestyle='--')
plt.plot(dataY_true[5201:],c='b', linestyle='--')
plt.suptitle('Training Traffic Flow Prediction')
plt.show()



# 测试集归一化处理
test_data = sc.fit_transform(test_set)

# 测试集切分
test_x,test_y = sliding_windows(test_data,n_steps)

test_x = Variable(torch.Tensor(test_x))
test_y = Variable(torch.Tensor(test_y))

# 利用训练好的模型对测试集进行预测
lstm.eval()
test_predict = lstm(test_x)

# 查看训练损失
loss_predict= criterion(test_predict,test_y)
print("prediction loss:%1.5f" % loss_predict.item())

# 将预测结果和真实数据结果转换为numpy的格式
test_predict = test_predict.data.numpy()
testY_true = test_y.data.numpy()

# 将预测结果和真实数据转换为原始数据范围
test_predict = sc.inverse_transform(test_predict)
testY_true = sc.inverse_transform(testY_true)


# 回归模型的各个指标评价：MSE,MAE,r2,MAPE
eva_regress(testY_true, test_predict)


y_preds = []
test_predict2 = test_predict.reshape(1,-1)[0]   #转换为行数据
testY_true2 = testY_true.reshape(1,-1)[0]       #转换为行数据

y_preds.append(test_predict2[0:288])

plot_results(testY_true2[:288], y_preds, names='LSTM')


