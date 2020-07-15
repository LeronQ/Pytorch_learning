 
import torch.nn as nn
import torch
 
# Training data
x_data = torch.tensor([[1.0,2.,3.], [2.0,4.,8.], [3.0,6.,9.], [4.0,8.,12.]])
y_data = torch.tensor([[0.], [0.], [1.], [1.]])
 
 
class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.linear = nn.Linear(3,1)  # three in and one out
 
    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred
 
model = Model()
 
learning_rate = 0.01
num_epochs = 1000
 
criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(),lr =learning_rate)
 
for epoch in range(num_epochs):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x_data)
 
    # Compute and print loss
    loss = criterion(y_pred,y_data)
 
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
 
 
y_pred = model(torch.tensor([[2.5,5.,7.5]]))
 
print(f'Prediction after [2,1] hour of training: {y_pred.item():.4f} | Above 50%: {y_pred.item() > 0.5}')
 
print(type(y_pred))
print('prediction:',y_pred.data)
