import MyD2l as d2l
import torch
from torch import nn


def conv_block(in_channels, num_channels):
    blk = nn.Sequential(
        nn.BatchNorm2d(in_channels), nn.ReLU(),
        nn.Conv2d(in_channels, num_channels, kernel_size=3, padding=1)  # 长宽不变
    )
    return blk


def transition_block(in_channels, num_channels):
    blk = nn.Sequential(
        nn.BatchNorm2d(in_channels), nn.ReLU(),
        nn.Conv2d(in_channels, num_channels, kernel_size=1),  # 仅改变通道
        nn.AvgPool2d(kernel_size=2, stride=2)  # 长宽减半
    )
    return blk


class DenseBlock(nn.Module):
    def __init__(self, num_convs, in_channels, num_channels):
        super().__init__()
        self.net = nn.Sequential()
        for i in range(num_convs):
            self.net.add_module('conv%d' % i,
                                conv_block(in_channels, num_channels))
            in_channels += num_channels

    def forward(self, X):
        for blk in self.net.children():
            Y = blk(X)
            X = torch.cat((X, Y), dim=1)
        return X


class Flatten(nn.Module):
    def __init__(self, flatten):
        super().__init__()
        self.flatten = flatten

    def forward(self, X):
        return X.view(-1, self.flatten)


class DenseNet(nn.Module):
    def __init__(self, num_channels, growth_rate, num_convs_in_dense_blocks, in_channels=1):
        super().__init__()
        self.first_conv = nn.Sequential(
            nn.Conv2d(in_channels, num_channels, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(num_channels), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.middle_dense = nn.Sequential()
        for i, num_convs in enumerate(num_convs_in_dense_blocks):
            self.middle_dense.add_module('dense%d' % i, DenseBlock(num_convs, num_channels, growth_rate))
            num_channels += num_convs * growth_rate
            if i != len(num_convs_in_dense_blocks) - 1:
                self.middle_dense.add_module('transition%d' % i,
                                             transition_block(num_channels, num_channels // 2))
                num_channels //= 2

        self.last_linear = nn.Sequential(
            nn.BatchNorm2d(num_channels), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
            Flatten(num_channels), nn.Linear(num_channels, 10)
        )

    def forward(self, X):
        X = self.first_conv(X)
        X = self.middle_dense(X)
        return self.last_linear(X)


if __name__ == '__main__':
    # blk = DenseBlock(2, 3, 10)
    # X = torch.rand(4, 3, 8, 8)
    # Y = blk(X)
    # print(Y.shape)
    # blk = transition_block(23, 10)
    # Y = blk(Y)
    # print(Y.shape)
    lr, num_epochs, batch_size, device = 0.1, 5, 256, torch.device('cuda')
    num_channels, growth_rate = 64, 32
    num_convs_in_dense_blocks = [4, 4, 4, 4]
    net = DenseNet(num_channels, growth_rate, num_convs_in_dense_blocks).to(device)
    d2l.initial(net)
    trainer = torch.optim.SGD(net.parameters(), lr=lr)
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
    d2l.train_ch5(net, train_iter, test_iter, trainer, num_epochs, device)
