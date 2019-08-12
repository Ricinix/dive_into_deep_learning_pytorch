import MyD2l as d2l
import torch
from torch import nn
from typing import Union


def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum, training):
    if not training:
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # 全连接层：算出每个样本的均值和方差
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            # 卷积层：算出每个通道的均值和方差
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        X_hat = (X - mean) / torch.sqrt(var + eps)

        # 更新（学习）移动平均均值和方差
        moving_mean = moving_mean * momentum + mean * (1.0 - momentum)
        moving_var = moving_var * momentum + var * (1.0 - momentum)
    Y = gamma * X_hat + beta
    return Y, moving_mean, moving_var


class BatchNorm(nn.Module):
    def __init__(self, num_features, num_dims):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)

        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))

        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.zeros(shape)

    def forward(self, X):
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma, self.beta, self.moving_mean,
            self.moving_var, eps=1e-5, momentum=0.9, training=self.training)
        return Y


class NLeNet(nn.Module):

    def __init__(self, in_shape: Union[tuple, list], in_channels=1):
        super().__init__()
        X_test = torch.rand(1, in_channels, *in_shape)
        self.conv_part = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5),
            BatchNorm(6, num_dims=4), nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            BatchNorm(16, num_dims=4), nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        X_test = self.conv_part(X_test)
        self.flatten = X_test.shape[1] * X_test.shape[2] * X_test.shape[3]
        self.linear_part = nn.Sequential(
            nn.Linear(self.flatten, 120),
            BatchNorm(120, num_dims=2), nn.Sigmoid(),
            nn.Linear(120, 84),
            BatchNorm(84, num_dims=2), nn.Sigmoid(),
            nn.Linear(84, 10)
        )

    def forward(self, X):
        X = self.conv_part(X)
        return self.linear_part(X.view(-1, self.flatten))


if __name__ == '__main__':
    lr, num_epochs, batch_size, device = 1.0, 5, 256, torch.device("cuda")
    net = NLeNet((28, 28))
    d2l.initial(net)
    trainer = torch.optim.SGD(net.parameters(), lr=lr)
    train_tier, test_iter = d2l.load_data_fashion_mnist(batch_size)
    d2l.train_ch5(net, train_tier, test_iter, trainer, num_epochs)
    for layer in net.modules():
        if isinstance(layer, BatchNorm):
            print(layer.gamma.view(-1,), layer.beta.view(-1,), sep='\n')
            break
