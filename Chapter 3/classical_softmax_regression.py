from time import time

import torch
from matplotlib import pyplot as plt

import MyD2l as d2l

num_inputs = 784
num_outputs = 10

W = torch.normal(mean=torch.zeros(size=(num_inputs, num_outputs)), std=0.01)
W.requires_grad_(True)
b = torch.zeros(num_outputs, requires_grad=True)


def softmax(X):
    X_exp = torch.exp(X)
    partition = torch.sum(X_exp, dim=1, keepdim=True)
    return X_exp / partition


def net(X):
    return softmax(torch.mm(X.view((-1, num_inputs)), W) + b)


def cross_entropy(y_hat, y):
    # y_pick = torch.zeros(y.shape[0])
    # for i in range(y_hat.shape[0]):
    #     y_pick[i] = y_hat[i, y[i].item()]
    # return -y_pick.log()
    y_pick = (y == 0).view(-1, 1)
    for j in range(1, y_hat.shape[1]):
        y_pick = torch.cat((y_pick, (y == j).view(-1, 1)), 1)
    return -torch.masked_select(y_hat, y_pick).log()


def accuracy(y_hat, y):
    acc = y_hat.argmax(dim=1) == y.type(torch.long)
    acc = acc.type(torch.float)
    return acc.mean().item()


if __name__ == '__main__':
    # y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
    # y = torch.tensor([0, 2], dtype=torch.int)
    # print(evaluate_accuracy(y_hat, y))
    num_epochs, lr = 5, 0.1
    train_iter, test_iter = d2l.data_load()
    now = time()
    d2l.train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs,
                  [W, b], lr)
    print("所需时间 %.4f" % (time() - now))
    for X, y in test_iter:
        break
    true_labels = d2l.get_fashion_mnist_labels(y.numpy())
    pred_labels = d2l.get_fashion_mnist_labels(net(X).argmax(dim=1).numpy())
    # 上面是正确标签，下面是预测标签
    titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]
    d2l.show_fashion_mnist(X[0:9], titles[0:9])
    plt.show()



