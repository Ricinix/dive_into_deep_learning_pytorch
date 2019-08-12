import torch
from torch import nn


def corr2d(X, K):
    h, w =K.shape
    Y = torch.zeros(X.shape[0] - h + 1, X.shape[1] - w + 1)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i: i + h, j: j + w] * K).sum()
    return Y


if __name__ == '__main__':
    # X = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    # K = torch.tensor([[0, 1], [2, 3]])
    # print(corr2d(X, K))
    X = torch.ones(6, 8)
    X[:, 2:6] = 0
    K = torch.tensor([[1., -1]])
    Y = corr2d(X, K)

    X = X.view(1, 1, 6, 8)
    Y = Y.view(1, 1, 6, 7)
    conv2d = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)

    for i in range(10):
        Y_hat = conv2d(X)
        l = (Y_hat - Y) ** 2
        l = l.sum()
        l.backward()
        with torch.no_grad():
            conv2d.weight -= 3e-2 * conv2d.weight.grad
            conv2d.weight.grad.zero_()
        print('batch %d, loss %.3f' % (i + 1, l.item()))
    print(conv2d.weight.view(1, 2))

