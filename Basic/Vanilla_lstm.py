from numpy import array
import torch.nn as nn
import torch
from torch.autograd import Variable
from sklearn.preprocessing import  MinMaxScaler
 
# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)
 
# define input sequence
raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
sc = MinMaxScaler()
raw_seq = sc.fit_transform(array(raw_seq).reshape(-1,1))
print(raw_seq)
 
'''
# choose a number of time steps
n_steps = 3
# split into samples
X, y = split_sequence(raw_seq, n_steps)
print(X,y)
# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))
print(X.shape)
print(X)
trainX = torch.Tensor(X)
trainY = torch.Tensor(y)
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
input_size = 1  # 输入维度
hidden_size = 128  # 隐层维度
num_layers = 1    # LSTM层数
num_classes = 1  # 输出维度
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
# Testing data
lstm.eval()
# Testing input data
x_input = array([70, 80, 90])
x_input = sc.fit_transform(array(x_input).reshape(-1,1))
x_input = x_input.reshape((1, n_steps, n_features))
X_input = Variable(torch.Tensor(x_input))
#
train_predict = lstm(X_input)
data_predict = sc.inverse_transform(train_predict.data.numpy())
print(data_predict)
'''
