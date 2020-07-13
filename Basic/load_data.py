
# coding: utf-8

# In[ ]:



'''
    1:数据
    2：模型
    3：损失
    4: 更新
'''

import torch 
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn

class DiabetesDataset(Dataset):
    def __ini__(self,filepath):
        xy = np.loadtxt(filepath,delimiter=',',dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:,:-1])
        self.y_data = torch.from_numpy(xy[:,[-1]])
        
    def __getitem__(self,index):
        return self.x_data[index],self.y_data[index]
    def __len__(self):
        return self.len
    
dataset = DiabetesDataset('diabetes.csv.gz')
# 常用的几个参数num_workers 表示进程数量
train_loader = DataLoader(dataset=dataset,batch_size=32,shuffle=True,num_workers=2)


class Model(nn.Module):
    def __ini__(self):
        super(Model,self).__ini__()
        self.linear1 = nn.Liner(8,6)
        self.linear2 = nn.Liner(6,4)
        self.linear3 = nn.Liner(4,1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

 if__name__=='__main__':
        for epoch in range(100):
            for i,(inputs,labels) in enumerate(train_loader):
                inputs,labels = data
                y_pred = model(inputs)
                # Forward
                loss= criterion(y_pred,labels)
                print(epoch,i,loss.item())
                # Backward
                optimizer.zero_grad()
                loss.backward()
                
                #update
                optimizer.step()


# In[ ]:



                

