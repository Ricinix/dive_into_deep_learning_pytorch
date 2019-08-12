import pickle
import random
from pathlib import Path
from time import time

import matplotlib.pyplot as plt
import torch
from IPython import display
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch import nn


# 打印
def show_fashion_mnist(images, labels):
    display.set_matplotlib_formats('svg')
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.reshape((28, 28)).asnumpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)


# 获取标签
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


# 平方损失
def square_loss(y_hat, y):
    return torch.sum((y_hat - y.reshape(y_hat.shape)).pow(2) / 2)


# 线性模型
def linreg(X, w, b):
    return torch.mm(X, w) + b


# 数据迭代
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = torch.tensor(indices[i: min(i + batch_size, num_examples)])
        yield features[j], labels[j]


# 设置svg
def use_svg_display():
    display.set_matplotlib_formats('svg')


# 设置图大小
def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize


def data_load():
    with open(Path('../data/torch_FashionMNIST_train'), 'rb') as f:
        train_iter = pickle.load(f)
    with open(Path('../data/torch_FashionMNIST_test'), 'rb') as f:
        test_iter = pickle.load(f)
    return train_iter, test_iter


def evaluate_accuracy(data_iter, net, input_num=None, device=None):
    acc_num, n = 0.0, 0
    for X, y in data_iter:
        if device is not None:
            X = X.cuda()
            y = y.cuda()
        if input_num is not None:
            X = X.view(-1, input_num)
        y = y.type(torch.long)
        acc_num += (net(X).argmax(dim=1) == y).sum().item()
        n += y.size(0)
    return acc_num / n


def sgd(params, lr):
    for param in params:
        param -= lr * param.grad
        param.grad.zero_()


def train_ch3(net, train_iter, test_iter, loss, num_epochs,
              params=None, lr=None, trainer=None, input_num=None):
    start_time = time()
    loss_time = 0.0
    backward_time = 0.0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            if input_num is not None:
                X = X.view(-1, input_num)

            if trainer is not None:
                trainer.zero_grad()

            y_hat = net(X)

            btime = time()
            l = loss(y_hat, y.type(torch.long)).mean()
            loss_time += time() - btime
            btime = time()
            l.backward()
            backward_time += time() - btime

            if trainer is None:
                with torch.no_grad():
                    sgd(params, lr)
            else:
                trainer.step()
            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y.type(torch.long)).sum().item()
            n += y.size(0)
        test_acc = evaluate_accuracy(test_iter, net, input_num=input_num)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))
    print("loss计算所需时间：%.5f" % loss_time)
    print("backward计算所需时间：%.5f" % backward_time)
    print("训练总耗时： %.5f" % (time() - start_time))


def train_ch5(net, train_iter, test_iter, trainer, num_epochs, device=None):
    print("初始化完成， 开始训练！")
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
        test_acc = evaluate_accuracy(test_iter, net, device=device)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc,
                 time() - start))


def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None,
            legend=None, figsize=(3.5, 2.5)):
    set_figsize(figsize)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        plt.semilogy(x2_vals, y2_vals, linestyle=':')
        plt.legend(legend)


def load_data_fashion_mnist(batch_size, resize=None, download=False, root='../data/fashion-mnist'):
    root = Path(root)
    transformer = []
    if resize:
        transformer += [transforms.Resize(resize)]
    transformer += [transforms.ToTensor()]
    transformer = transforms.Compose(transformer)
    mnist_train = datasets.FashionMNIST(root=root, train=True, transform=transformer, download=download)
    mnist_test = datasets.FashionMNIST(root=root, train=False, transform=transformer, download=download)
    train_iter = DataLoader(mnist_train, batch_size, shuffle=True, num_workers=4)
    test_iter = DataLoader(mnist_test, batch_size, shuffle=True, num_workers=4)
    return train_iter, test_iter


def initial(net):
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)


class Flatten(nn.Module):
    def __init__(self, flatten):
        super().__init__()
        self.flatten = flatten

    def forward(self, X):
        return X.view(-1, self.flatten)

