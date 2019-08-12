import MyD2l as d2l
import torch
from torch import nn


class Flatten(nn.Module):

    def forward(self, X):
        return X.view(-1, 10)


def nin_block(num_channels, kernel_size, strides, padding, in_channels=1):
    blk = nn.Sequential(
        nn.Conv2d(in_channels, num_channels, kernel_size, strides, padding), nn.ReLU(),
        # nn.Conv2d(num_channels, num_channels, 1), nn.ReLU(),
        nn.Conv2d(num_channels, num_channels, 1), nn.ReLU()
    )
    return blk


net = nn.Sequential(
        nin_block(96, kernel_size=11, strides=4, padding=0),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nin_block(256, kernel_size=5, strides=1, padding=2, in_channels=96),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nin_block(384, kernel_size=3, strides=1, padding=1, in_channels=256),
        nn.MaxPool2d(kernel_size=3, stride=2), nn.Dropout(0.5),

        nin_block(10, kernel_size=3, strides=1, padding=1, in_channels=384),
        nn.AdaptiveAvgPool2d(1),
        Flatten()
    ).to(torch.device('cuda'))

if __name__ == '__main__':
    # X = torch.rand(1, 1, 224, 224)
    # for layer in net.children():
    #     X = layer(X)
    #     print(layer, '\noutput shape:\t', X.shape, '\n')
    lr, num_epochs, batch_size, device = 0.1, 5, 128, torch.device('cuda')
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    trainer = torch.optim.SGD(net.parameters(), lr=lr)
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
    d2l.train_ch5(net, train_iter, test_iter, trainer, num_epochs, device)
