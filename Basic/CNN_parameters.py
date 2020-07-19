
# coding: utf-8

# In[1]:


import torch

in_channels,out_channels =5,10
width,heigth = 100,100
kernel_size = 3
batch_size = 1

input = torch.randn(batch_size,in_channels,width,heigth)
conv_layer = torch.nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size)

out_put = conv_layer(input)

print(input.shape)
print(out_put.shape)
print(conv_layer.weight.shape)

