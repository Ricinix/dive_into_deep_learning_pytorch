import torch
from torch import nn


if __name__ == '__main__':
    net = nn.Sequential()
    net.add_module('Linear 1', nn.Linear(20, 256, bias=True))
    net.add_module('Linear 2', nn.Linear(256, 10, True))

    print(net[0].weight)
    nn.init.normal_(net[0].weight)
    nn.init.constant_(net[0].weight, 0)
    print(net.state_dict())
    print(net[0].weight.data)
    print(net[0].parameters())
    X = torch.rand(2, 20)
    Y = net(X)
    print(Y)
