
# coding: utf-8

# In[2]:


import torch
import torch.nn as nn
import torch.nn.functional as F

'''
用两个inception 模块实现googleNet
'''

#将inception 抽象成为一个类
class InceptionA(nn.Module):
    def __init__(self,in_channels):
        super(InceptionA,self).__init__():
        self.branch1x1 = nn.Conv2d(in_channels,16,kernel_size=1)
        
        self.branch5x5_1 = nn.Conv2d(in_channels,16,kernel_size=1)
        self.branch5x5_2 = nn.Conv2d(16,24,kernel_size=5,padding=2)
        
        self.branch3x3_1 = nn.Conv2d(in_channels,16,kernel_size=1)
        self.branch3x3_2 = nn.Conv2d(16,24,kernel_size=3,padding=1)
        self.branch3x3_3 = nn.Conv2d(24,24,kernel_size=3,padding=1)
        
        self.branch_pool = nn.Conv2d(in_channels,24,kernel_size=1)
        
    def forward(self,x):
        branch1x1 = self.branch1x1(x)
        
        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)
        
        branch_pool = F.avg_pool2d(x,kernel_size=2,stride=1,padding=1)
        branch_pool = self.branch_pool(branch_pool)
        
        # 输出格式[batch_size,channel,width,height]
        outputs = [branch1x1,branch5x5,branch3x3,branch_pool]
        
        # 按照通道数进出拼接
        return torch.cat(outputs,dim=1)
        
        
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(1,10,kernel_size=5)
        self.conv2 = nn.Conv2d(88,20,kernel_size=5)
        
        self.incep1 = InceptionA(in_channels=10)
        self.incep2 = InceptionA(in_channels=20)
        
        self.mp = nn.MaxPool2d(2)
        self.fc = nn.Linear(1408,10)
        
    def forward(self,x):
        in_size = x.sixe(0)
        x = F.relu(self.mp(self.conv1(x)))
        x = self.incep1(x)
        x = F.relu(self.mp(self.conv2(x)))
        x = self.incep2(x)
        x = x.view(in_size,-1)
        x = self.fc(x)
        
        return x
    
    

