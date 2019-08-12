import math

import torch
from torch import nn

import MyD2l as d2l


def calculate_shape(shape, *k_p_s):
    if shape[0] == shape[1]:
        for k, p, s in k_p_s:
            shape[0] = math.floor((shape[0] - k + p * 2 + s) / s)
        return shape[0], shape[0]
    else:
        for k, p, s in k_p_s:
            shape[0] = math.floor((shape[0] - k + p * 2 + s) / s)
        for k, p, s in k_p_s:
            shape[1] = math.floor((shape[1] - k + p * 2 + s) / s)
        return shape[0], shape[1]


class AlexNet(nn.Module):

    def __init__(self, shape):
        super().__init__()
        shape = [shape[-2], shape[-1]]
        self.conv_pool_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.pure_conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        shape = calculate_shape(shape, (11, 0, 4), (3, 0, 2), (5, 2, 1), (3, 0, 2),
                                (3, 1, 1), (3, 1, 1), (3, 1, 1), (3, 0, 2))
        self.linear_layer = nn.Sequential(
            nn.Linear(in_features=256 * shape[0] * shape[1], out_features=4096),
            nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(in_features=4096, out_features=10)
        )

    def forward(self, X):
        out = self.conv_pool_layer(X)
        out = self.pure_conv_layer(out)
        return self.linear_layer(out.view(-1, 6400))


if __name__ == '__main__':
    # X = torch.rand(1, 1, 224, 224)
    # net = AlexNet(X.shape)
    # i = 0
    # for layers in net.children():
    #     for layer in layers:
    #         X = layer(X)
    #         print(layer, '\noutput shape:\t', X.shape)
    #     i += 1
    #     if i == 2:
    #         X = X.view(-1, 6400)
    batch_size = 128
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, 224)

    lr, num_epochs, device = 0.01, 15, torch.device('cuda')
    net = AlexNet((224, 224)).to(device)
    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    trainer = torch.optim.SGD(net.parameters(), lr=lr)
    d2l.train_ch5(net, train_iter, test_iter, trainer, num_epochs, device)
