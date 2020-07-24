import torch
from torch.autograd import Variable
from torch import nn
 
# 输入维度=28
# 输出维度=4
# time_steps=3
# batch_size=True
# 数据量大小=10
# 隐层神经元个数=4
# 网络层次数量=1
 
import torch
from torch.autograd import Variable
from torch import nn
 
'''
    batch_first = False :   输入形式：(seq,batch,feature)
'''
num_layers = 1
lstm_seq = nn.LSTM(28, 4, num_layers=num_layers,batch_first=False)  # 构建LSTM网络
lstm_input = Variable(torch.randn(10, 3, 28))  # 构建输入
h_init = Variable(torch.randn(1, 3, 4))  # 构建h输入参数
c_init = Variable(torch.randn(1, 3, 4))  # 构建c输出参数
out, (h, c) = lstm_seq(lstm_input, (h_init, c_init))  # 计算
print(lstm_seq.weight_ih_l0.shape)
print(lstm_seq.weight_hh_l0.shape)
print(out.shape, h.shape, c.shape)
 
 
print('-----other model------')
 
'''
    batch_first = True :   输入形式：(batch, seq, feature)
'''
lstm_seq = nn.LSTM(28, 4, num_layers=1,batch_first=True)  # 构建LSTM网络
lstm_input = Variable(torch.randn(10, 3, 28))  # 构建输入
h_init = Variable(torch.randn(1, 10, 4))  # 构建h输入参数   -- 每个batch对应最后一个time_steps的隐层,也就是输出的结果值对应的隐层
c_init = Variable(torch.randn(1, 10, 4))  # 构建c输出参数   -- 每个batch对应最后一个time_steps的隐层，也就是输出的结果值对应的隐层
out, (h, c) = lstm_seq(lstm_input, (h_init, c_init))  # 计算
print(lstm_seq.weight_ih_l0.shape)
print(lstm_seq.weight_hh_l0.shape)
print(out.shape, h.shape, c.shape)
 
print('-----other model------')
 
'''
    batch_first = True :   输入形式：(batch, seq, feature)
    bidirectional = True
'''
num_layers = 1
bidirectional_set  = True
bidirectional = 1 if bidirectional_set else 0
 
 
lstm_seq = nn.LSTM(28, 4, num_layers=num_layers,bidirectional=bidirectional_set,batch_first=False)  # 构建LSTM网络
lstm_input = Variable(torch.randn(10, 3, 28))  # 构建输入
h_init = Variable(torch.randn(2*num_layers*bidirectional, 3, 4))  # 构建h输入参数
c_init = Variable(torch.randn(2*num_layers*bidirectional, 3, 4))  # 构建c输出参数
out, (h, c) = lstm_seq(lstm_input, (h_init, c_init))  # 计算
print(lstm_seq.weight_ih_l0.shape)
print(lstm_seq.weight_hh_l0.shape)
print(out.shape, h.shape, c.shape)
 
 
 
