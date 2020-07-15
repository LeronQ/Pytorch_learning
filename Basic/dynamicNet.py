 
import torch
import random
import torch.nn as nn
 
class DynamicNet(nn.Module):
    def __init__(self,D_in,H,D_out):
        super(DynamicNet,self).__init__()
        self.input_layer = nn.Linear(D_in,H)
        self.middle_layer = nn.Linear(H,H)
        self.out_layer = nn.Linear(H,D_out)
 
    def forward(self, x):
        h_relu = self.input_layer(x).clamp(min=0)
        for _ in range(random.randint(0,3)):
            # Dynamic Net
            h_relu = self.middle_layer(h_relu).clamp(min=0)
        y_pred = self.out_layer(h_relu)
        return y_pred
 
N, D_in, H, D_out = 64, 1000, 100, 10
 
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)
 
model = DynamicNet(D_in,H,D_out)
 
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(),lr = 1e-4, momentum=0.9)
 
for t in range(500):
    y_pred = model(x)
 
    loss = criterion(y_pred,y)
    if t%100==99:
        print(t,loss.item())
 
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
 
 
 
 
