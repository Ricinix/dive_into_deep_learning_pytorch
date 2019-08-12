import torch
import torch.nn as nn
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import pickle
from pathlib import Path
from time import time

num_inputs = 784
num_outputs = 10


def data_load():
    with open(Path('../data/torch_FashionMNIST_train'), 'rb') as f:
        train_iter = pickle.load(f)
    with open(Path('../data/torch_FashionMNIST_test'), 'rb') as f:
        test_iter = pickle.load(f)
    return train_iter, test_iter


class SoftmaxRegression(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.Linear = nn.Linear(in_features, out_features, bias=True)

    def forward(self, X):
        return self.Linear(X.view(-1, self.in_features))


def evaluate_accuracy(data_iter, net):
    acc_num, n = 0.0, 0
    for X, y in data_iter:
        y = y.type(torch.long)
        acc_num += (net(X).argmax(dim=1) == y).sum().item()
        n += y.size(0)
    return acc_num / n


def train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer=None):
    loss_time, backward_time, test_acc = 0.0, 0.0, 0.0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            trainer.zero_grad()
            y_hat = net(X)
            btime = time()
            l = loss(y_hat, y.type(torch.long)).sum()
            loss_time += time() - btime
            btime = time()
            l.backward()
            backward_time += time() - btime
            trainer.step()
            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y.type(torch.long)).sum().item()
            n += y.size(0)
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))
    print("loss计算所需时间：%.5f" % loss_time)
    print("backward计算所需时间：%.5f" % backward_time)
    return test_acc


def test_performance(module, train_iter, test_iter, loss):
    epochs = 8
    lrs = [0.01, 0.05, 0.1, 0.2, 0.4]
    df = pd.DataFrame(columns=("epoch", "lr", "test_acc"))
    for epoch in range(1, epochs + 1):
        for lr in lrs:
            nn.init.normal_(module.Linear.weight, 0, 0.01)
            nn.init.normal_(module.Linear.bias, 0, 0.01)
            test_acc = train_ch3(module, train_iter, test_iter, loss, epoch,
                                 trainer=torch.optim.SGD(module.parameters(), lr=lr))
            df = df.append({'epoch': epoch, 'lr': lr, 'test_acc': test_acc}, ignore_index=True)
    with open(Path('../data/performance.pickle'), 'wb') as f:
        pickle.dump(df, f)
    sns.relplot(x="epoch", y="test_acc", hue="lr", data=df, kind="line")
    plt.show()


if __name__ == '__main__':
    train_iter, test_iter = data_load()
    module = SoftmaxRegression(num_inputs, num_outputs)
    nn.init.normal_(module.Linear.weight, 0, 0.01)
    loss = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    # num_epochs = 5
    # btime = time()
    # train_ch3(module, train_iter, test_iter, loss, num_epochs, trainer=optimizer)
    # print("训练耗时: %.5f" % (time() - btime))
    test_performance(module, train_iter, test_iter, loss)
