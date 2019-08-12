import torch
from torch import nn

import MyD2l as d2l

num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = torch.normal(mean=torch.zeros(size=(num_inputs, num_hiddens)), std=0.01)
b1 = torch.zeros(num_hiddens)
W2 = torch.normal(mean=torch.zeros(size=(num_hiddens, num_outputs)), std=0.01)
b2 = torch.zeros(num_outputs)

loss = nn.CrossEntropyLoss()


def relu(X):
    return X.clamp(min=0)


def net(X):
    H = relu(torch.mm(X.view(-1, num_inputs), W1) + b1)
    return torch.mm(H, W2) + b2


if __name__ == '__main__':
    params = [W1, b1, W2, b2]
    for param in params:
        param.requires_grad_(True)
    train_iter, test_iter = d2l.data_load()
    num_epochs, lr = 5, 0.5
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, 256, params, lr)