import torch
import MyD2l as d2l
from torch import nn
from time import time

num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
drop_prob1, drop_prob2 = 0.2, 0.5
num_epochs, lr, batch_size = 5, 0.5, 256

W1 = torch.normal(mean=torch.zeros(num_inputs, num_hiddens1), std=0.01)
b1 = torch.zeros(num_hiddens1)
W2 = torch.normal(mean=torch.zeros(num_hiddens1, num_hiddens2), std=0.01)
b2 = torch.zeros(num_hiddens2)
W3 = torch.normal(mean=torch.zeros(num_hiddens2, num_outputs), std=0.01)
b3 = torch.zeros(num_outputs)

params = [W1, b1, W2, b2, W3, b3]
for param in params:
    param.requires_grad_(True)


def dropout(X, drop_prob):
    assert 0 <= drop_prob <= 1
    keep_prob = 1 - drop_prob

    if keep_prob == 0:
        return torch.zeros_like(X)
    mask = torch.rand(size=X.shape) < keep_prob
    return mask.type(torch.float) * X / keep_prob


def net(X, is_train):
    X = X.reshape((-1, num_inputs))
    H1 = (torch.mm(X, W1) + b1).relu()
    if is_train:
        H1 = dropout(H1, drop_prob1)
    H2 = (torch.mm(H1, W2) + b2).relu()
    if is_train:
        H2 = dropout(H2, drop_prob2)
    return torch.mm(H2, W3) + b3


def evaluate_accuracy(data_iter, net):
    acc_num, n = 0.0, 0
    for X, y in data_iter:
        y = y.type(torch.long)
        acc_num += (net(X, False).argmax(dim=1) == y).sum().item()
        n += y.size(0)
    return acc_num / n


def sgd(params, lr):
    for param in params:
        param -= lr * param.grad
        param.grad.zero_()


def train_ch3(net, train_iter, test_iter, loss, num_epochs,
              params=None, lr=None, trainer=None):
    loss_time = 0.0
    backward_time = 0.0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            if trainer is not None:
                trainer.zero_grad()
            y_hat = net(X, True)

            btime = time()
            l = loss(y_hat, y.type(torch.long)).mean()
            loss_time += time() - btime
            btime = time()
            l.backward()
            backward_time += time() - btime

            if trainer is None:
                with torch.no_grad():
                    sgd(params, lr)
            else:
                trainer.step()
            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y.type(torch.long)).sum().item()
            n += y.size(0)
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))
    print("loss计算所需时间：%.5f" % loss_time)
    print("backward计算所需时间：%.5f" % backward_time)


if __name__ == '__main__':
    loss = nn.CrossEntropyLoss()
    train_iter, test_iter = d2l.data_load()
    train_ch3(net, train_iter, test_iter, loss, num_epochs, params, lr)
