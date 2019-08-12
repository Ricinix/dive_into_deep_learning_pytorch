import MyD2l as d2l
import torch
from torch import nn
from torch.nn import functional as F


class Residual(nn.Module):
    def __init__(self, in_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, num_channels, kernel_size=3, padding=1,
                               stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, num_channels, kernel_size=1,
                                   stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)


class Flatten(nn.Module):
    def forward(self, X):
        return X.view(-1, 512)


def resnet_block(in_channels, num_channels, num_residuals, first_block=False):
    blk = nn.Sequential()
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.add_module('r%d' % i, Residual(in_channels, num_channels,
                                               use_1x1conv=True, strides=2))
        else:
            blk.add_module('r%d' % i, Residual(in_channels, num_channels))
        in_channels = num_channels
    return blk


if __name__ == '__main__':
    # blk = Residual(3, 6, True, 2)
    # X = torch.rand(4, 3, 6, 6)
    # print(blk(X).shape)
    net = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(64), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
    net.add_module('res0', resnet_block(64, 64, 2, first_block=True))
    net.add_module('res1', resnet_block(64, 128, 2))
    net.add_module('res2', resnet_block(128, 256, 2))
    net.add_module('res3', resnet_block(256, 512, 2))
    net.add_module('avgpool', nn.AdaptiveAvgPool2d(1))
    net.add_module('flatten', Flatten())
    net.add_module('linear', nn.Linear(512, 10))

    # X = torch.rand(1, 1, 224, 224)
    # for layer in net.children():
    #     X = layer(X)
    #     print('output shape:\t', X.shape, '\n')
    lr, num_epochs, batch_size, device = 0.05, 5, 256, torch.device('cuda')
    d2l.initial(net)
    net = net.to(device)
    trainer = torch.optim.SGD(net.parameters(), lr=lr)
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
    d2l.train_ch5(net, train_iter, test_iter, trainer, num_epochs, device)

