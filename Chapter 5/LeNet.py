import torch
import MyD2l as d2l
from time import time
from torch import nn


class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5), nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5), nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer1 = nn.Sequential(
            nn.Linear(256, 120), nn.Sigmoid(),
            nn.Linear(120, 84), nn.Sigmoid(),
            nn.Linear(84, 10)
        )

    def forward(self, X):
        X = self.layer0(X)
        return self.layer1(X.view(-1, 256))


def train_ch5(net, train_iter, test_iter, trainer, num_epochs, device=None):
    loss = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time()
        for X, y in train_iter:
            if device is not None:
                X = X.to(device)
                y = y.to(device)
            trainer.zero_grad()
            y_hat = net(X)
            l = loss(y_hat, y.type(torch.long))
            l.backward()
            trainer.step()
            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y.type(torch.long)).sum().item()
            n += y.size(0)
        test_acc = d2l.evaluate_accuracy(test_iter, net, device=device)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc,
                 time() - start))


if __name__ == '__main__':
    train_iter, test_iter = d2l.data_load()
    device = torch.device('cuda')

    net = LeNet().to(device)
    for layer in net.modules():
        if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
            print('正在初始化', layer)
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    trainer = torch.optim.SGD(net.parameters(), lr=0.9)
    train_ch5(net, train_iter, test_iter, trainer, 5, device)

