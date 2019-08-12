import MyD2l as d2l
import torch
from torch import nn
from matplotlib import pyplot as plt

class Flatten(nn.Module):
    def forward(self, X):
        n = 1
        for i in X.shape[1:]:
            n *= i
        return X.view(-1, n)


def vgg_block(num_convs, num_channels, in_channels):
    blk = nn.Sequential()
    blk.add_module('conv0', nn.Conv2d(in_channels=in_channels, out_channels=num_channels,
                                      kernel_size=3, padding=1))
    blk.add_module('relu0', nn.ReLU())
    for i in range(1, num_convs):
        blk.add_module('conv%d' % i, nn.Conv2d(in_channels=num_channels, out_channels=num_channels,
                                               kernel_size=3, padding=1))
        blk.add_module('relu%d' % i, nn.ReLU())
    blk.add_module('maxpool', nn.MaxPool2d(kernel_size=2, stride=2))
    return blk


def vgg(conv_arch, shape, in_channels=1):
    net = nn.Sequential()
    X_test = torch.rand(1, in_channels, *shape[-2:])
    i = 0
    for (num_convs, num_channels) in conv_arch:
        net.add_module('vgg%d' % i, vgg_block(num_convs, num_channels, in_channels))
        in_channels = num_channels
        i += 1
    X_test = net(X_test)
    flatten = X_test.shape[1] * X_test.shape[2] * X_test.shape[3]
    net.add_module('liear_part', nn.Sequential(Flatten(),
                                               nn.Linear(flatten, 4096),
                                               nn.ReLU(), nn.Dropout(0.5),
                                               nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
                                               nn.Linear(4096, 10)))
    return net


def test_resize(X, shape_n):
    plt.imshow(X.view(shape_n, shape_n).numpy())
    plt.show()


if __name__ == '__main__':
    conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
    # X = torch.rand(1, 1, 224, 224)
    # for blk in net.modules():
    #     if not isinstance(blk, nn.Sequential) and not isinstance(blk, nn.ReLU):
    #         X = blk(X)
    #         print(blk, '\noutput shape:\t', X.shape)

    ratio = 4
    device = torch.device('cuda')
    small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
    net = vgg(small_conv_arch, (96, 96)).to(device)
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    lr, num_epochs, batch_size = 0.05, 5, 128
    trainer = torch.optim.SGD(net.parameters(), lr=lr)
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
    d2l.train_ch5(net, train_iter, test_iter, trainer, num_epochs, device)
