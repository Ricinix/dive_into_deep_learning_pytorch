import MyD2l as d2l
import torch
from torch import nn


class Inception(nn.Module):
    def __init__(self, c_in, c1, c2, c3, c4):
        super().__init__()
        self.p1 = nn.Sequential(
            nn.Conv2d(c_in, c1, kernel_size=1), nn.ReLU()
        )
        self.p2 = nn.Sequential(
            nn.Conv2d(c_in, c2[0], kernel_size=1), nn.ReLU(),
            nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1), nn.ReLU()
        )
        self.p3 = nn.Sequential(
            nn.Conv2d(c_in, c3[0], kernel_size=1), nn.ReLU(),
            nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2), nn.ReLU()
        )
        self.p4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(c_in, c4, kernel_size=1), nn.ReLU()
        )

    def forward(self, X):
        p1 = self.p1(X)
        p2 = self.p2(X)
        p3 = self.p3(X)
        p4 = self.p4(X)
        return torch.cat((p1, p2, p3, p4), dim=1)


b1 = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)

b2 = nn.Sequential(
    nn.Conv2d(64, 64, kernel_size=1), nn.ReLU(),
    nn.Conv2d(64, 192, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)

b3 = nn.Sequential(
    Inception(192, 64, (96, 128), (16, 32), 32),
    Inception(256, 128, (128, 192), (32, 96), 64),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)

b4 = nn.Sequential(
    Inception(480, 192, (96, 208), (16, 48), 64),
    Inception(512, 160, (112, 224), (24, 64), 64),
    Inception(512, 128, (128, 256), (24, 64), 64),
    Inception(512, 112, (144, 288), (32, 64), 64),
    Inception(528, 256, (160, 320), (32, 128), 128),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)

b5 = nn.Sequential(
    Inception(832, 256, (160, 320), (32, 128), 128),
    Inception(832, 384, (192, 384), (48, 128), 128),
    nn.AdaptiveAvgPool2d(1)
)


class Flatten(nn.Module):
    def forward(self, X):
        flat = X.shape[1] * X.shape[2] * X.shape[3]
        return X.view(-1, flat)


if __name__ == '__main__':
    # X = torch.rand(1, 1, 96, 96)
    # for layer in net.children():
    #     X = layer(X)
    #     print('output shape:\t', X.shape, '\n')

    lr, num_epochs, batch_size, device = 0.1, 5, 128, torch.device('cuda')
    net = nn.Sequential(b1, b2, b3, b4, b5, Flatten(), nn.Linear(1024, 10)).to(device)
    d2l.initial(net)
    trainer = torch.optim.SGD(net.parameters(), lr=lr)
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
    d2l.train_ch5(net, train_iter, test_iter, trainer, num_epochs, device)


