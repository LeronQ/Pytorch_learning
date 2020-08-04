# @Time    : 2020/8/4 
# @Author  : LeronQ
# @github  : https://github.com/LeronQ


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
from sklearn.preprocessing import MinMaxScaler
import sklearn.metrics as metrics
import matplotlib as mpl
import math

from utils.plot_res import plot_results
from utils.eval_res import eva_regress
import warnings

warnings.filterwarnings("ignore")


def parse_args():
    '''
    :parameter setting
    :return:
    '''
    parser = argparse.ArgumentParser(description="PyTorch implementation of LSTM for traffic flow prediction")
    parser.add_argument('--batchsize', type=int, default=128, help='input batch size [128]')

    # LSTM layer parameters setting
    parser.add_argument('--hidden_size', type=int, default=64, help='size of hidden states [64, 128]')
    parser.add_argument('--num_layers', type=int, default=1, help='the number of hidden layers')

    parser.add_argument('--num_epochs', type=int, default=500, help='number of epochs to train [10, 200, 800]')
    parser.add_argument('--input_size', type=int, default=1, help='the number of input feature')
    parser.add_argument('--output_size', type=int, default=1, help='the number of output feature')
    parser.add_argument('--n_steps', type=int, default=12, help='the number of time steps in the window [12]')
    parser.add_argument('--learning_rate', type=int, default=0.01, help='the parameter of learning_rate')

    args = parser.parse_args()

    return args


def load_train_data():
    print("==> Loading train dataset ...")
    training_set = pd.read_csv('data/train.csv')
    training_set = training_set.iloc[:, 1:2].values
    return training_set


def load_test_data():
    print("==> Loading testing dataset ...")
    test_set = pd.read_csv('data/test.csv')
    test_set = test_set.iloc[:, 1:2].values

    return test_set


def process_data(data_set, n_steps):
    print("==> Processing dataset...")

    # data normalization
    sc = MinMaxScaler()
    scaler_data = sc.fit_transform(data_set)

    # using the 12 past data(one hour) to predict current traffic flow
    x, y = [], []
    for i in range(len(scaler_data)):
        end_ix = i + n_steps
        if end_ix > len(scaler_data) - 1:
            break
        seq_x = scaler_data[i:end_ix]
        seq_y = scaler_data[end_ix]
        x.append(seq_x)
        y.append(seq_y)
    return np.array(x), np.array(y), sc


# bulid model
class LSTM_Predict(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super(LSTM_Predict, self).__init__()
        self.output_size = output_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h_0 = Variable(torch.randn(self.num_layers, x.size(0),
                                   self.hidden_size)).to(device)
        c_0 = Variable(torch.randn(self.num_layers, x.size(0),
                                   self.hidden_size)).to(device)

        output, (h_out, _) = self.lstm(x, (h_0, c_0))
        h_out = h_out.view(-1, self.hidden_size)

        out = self.fc(h_out)
        return out


def main():
    """Main pipeline of vanilla-LSTM."""
    args = parse_args()

    training_set = load_train_data()

    x, y, sc = process_data(training_set, args.n_steps)
    train_size = int(len(x) * 0.67)

    # the total dataset x
    dataX = torch.Tensor(x).to(device)
    dataY = torch.Tensor(y).to(device)

    # split the x into train and validation dataset
    trainX = Variable(torch.Tensor(x[0:train_size])).to(device)
    trainY = Variable(torch.Tensor(y[0:train_size])).to(device)

    valdX = Variable(torch.Tensor(x[train_size:])).to(device)
    valdY = Variable(torch.Tensor(y[train_size:])).to(device)

    # Initialize model
    lstm = LSTM_Predict(args.input_size, args.output_size, args.hidden_size, args.num_layers)
    lstm.to(device)

    # MSE
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(lstm.parameters(), lr=args.learning_rate)
    loss_his = []

    print("==> Start training ...")
    for epoch in range(args.num_epochs):
        outputs = lstm(trainX)
        optimizer.zero_grad()

        loss = criterion(outputs, trainY)
        loss_his.append(loss.item())
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print("Epoch: %d, Loss: %1.5f" % (epoch, loss.item()))

    # save train model
    torch.save(lstm.state_dict(), 'lstm.pkl')

    # plot the loss of epochs
    plt.plot([i for i in range(args.num_epochs)], loss_his, '--', label='Predictions', alpha=0.5)
    plt.savefig('pic/epoch-loss.png')
    plt.show()

    # model prediction
    lstm.eval()
    train_predict = lstm(dataX)

    # output the predict result
    dataY_predict = train_predict.cpu().detach().numpy()
    dataY_true = dataY.data.cpu().detach().numpy()

    # convert the predict and true data to initial data scaler
    dataY_predict = sc.inverse_transform(dataY_predict)
    dataY_true = sc.inverse_transform(dataY_true)

    # plt.axvline(x=train_size, c='r', linestyle='--')
    plt.plot(dataY_predict[5201:], c='r', linestyle='--')
    plt.plot(dataY_true[5201:], c='b', linestyle='--')
    plt.suptitle('Training Traffic Flow Prediction')
    plt.savefig('pic/Training-Traffic-Flow-Prediction.png')
    plt.show()



def test():
    args = parse_args()
    test_set = load_test_data()
    test_x, test_y, sc = process_data(test_set, args.n_steps)

    lstm_pre_model = LSTM_Predict(args.input_size, args.output_size, args.hidden_size, args.num_layers)
    lstm_pre_model.to(device)

    # load the saved pkl
    lstm_pre_model.load_state_dict(torch.load('lstm.pkl'))

    test_x = Variable(torch.Tensor(test_x)).to(device)
    test_y = Variable(torch.Tensor(test_y)).to(device)

    # load model
    test_predict = lstm_pre_model(test_x)

    # check the loss of testing
    criterion = torch.nn.MSELoss()
    loss_predict = criterion(test_predict, test_y)
    print("Prediction loss:%1.5f" % loss_predict.item())

    # convert the tensor into numpy
    test_predict = test_predict.cpu().detach().numpy()
    testY_true = test_y.cpu().detach().numpy()

    # convert the predict and true data into initial data scaler
    test_predict = sc.inverse_transform(test_predict)
    testY_true = sc.inverse_transform(testY_true)

    # to evaluate the model：MSE,MAE r2 MAPE
    eva_regress(testY_true, test_predict)


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    main()
    test()

