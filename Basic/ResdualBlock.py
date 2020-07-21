
# coding: utf-8

# In[3]:


import torch
import torch.nn as nn
import torch.nn.functional as F


class ResdualBlock(nn.Module):
    def __init__(self,channels):
        super(ResdualBlock,self).__init__()
        self.channels = channels
        self.conv1 = nn.Conv2d(channels,channels,kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(channels.channels,kernel_size=3,padding=1)
        
    def forward(self,x):
        y = F.relu(self.conv1(x))
        y = self.conv2(y)
        
        return F.relu(x+y)
    
    
    

