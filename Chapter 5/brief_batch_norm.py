import MyD2l as d2l
import torch
from torch import nn


class NLeNet(nn.Module):

    def __init__(self, X_shape, in_channels=1):
        super().__init__()
        X_test = torch.rand(1, in_channels, *X_shape)
        self.conv_part = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5),
            nn.BatchNorm2d(6), nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.BatchNorm2d(16), nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        X_test = self.conv_part(X_test)
        self.flatten = X_test.shape[1] * X_test.shape[2] * X_test.shape[3]
        self.linear_part = nn.Sequential(
            nn.Linear(self.flatten, 120),
            nn.BatchNorm1d(120), nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.BatchNorm1d(84), nn.Sigmoid(),
            nn.Linear(84, 10)
        )

    def forward(self, X):
        X = self.conv_part(X)
        return self.linear_part(X.view(-1, self.flatten))


if __name__ == '__main__':
    lr, num_epochs, batch_size, device = 5.0, 5, 256, torch.device("cuda")
    net = NLeNet((28, 28)).to(device)
    trainer = torch.optim.SGD(net.parameters(), lr=lr)
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    d2l.train_ch5(net, train_iter, test_iter, trainer, num_epochs, device)
