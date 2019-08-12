import MyD2l as d2l
from torch import nn
import torch


class MLP(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens):
        super().__init__()
        self.num_inputs = num_inputs
        self.Linear1 = nn.Linear(num_inputs, num_hiddens, bias=True)
        self.ReLU = nn.ReLU()
        self.Linear2 = nn.Linear(num_hiddens, num_outputs, bias=True)
        self.Linear_hidden = nn.Linear(num_hiddens, num_hiddens, bias=True)

    def forward(self, X):
        H = self.ReLU(self.Linear1(X.view(-1, self.num_inputs)))
        H2 = self.ReLU(self.Linear_hidden(H))
        return self.Linear2(H2)


if __name__ == '__main__':
    num_inputs, num_outputs, num_hiddens = 784, 10, 256
    train_iter, test_iter = d2l.data_load()

    net = MLP(num_inputs, num_outputs, num_hiddens)
    for m in net.children():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.zeros_(m.bias)
    loss = nn.CrossEntropyLoss()
    trainer = torch.optim.SGD(net.parameters(), 0.5)

    num_epochs = 5
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, 256, trainer=trainer)