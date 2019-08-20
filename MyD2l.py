import math
import os
import pickle
import random
import zipfile
from pathlib import Path
from time import time

import matplotlib.pyplot as plt
import torch
from IPython import display
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets


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


def load_data_jay_lyrics():
    if 'deep_learning' in os.getcwd():
        path = Path('../data/jaychou_lyrics.txt.zip')
    else:
        path = Path('deep learning/data/jaychou_lyrics.txt.zip')
    with zipfile.ZipFile(path) as zin:
        with zin.open('jaychou_lyrics.txt') as f:
            corpus_chars = f.read().decode('utf-8')
    corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', '')
    corpus_chars = corpus_chars[:10000]
    idx_to_char = list(set(corpus_chars))
    char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
    vocab_size = len(char_to_idx)
    corpus_indices = [char_to_idx[char] for char in corpus_chars]
    return corpus_indices, char_to_idx, idx_to_char, vocab_size


def data_iter_random(corpus_indices, batch_size, num_steps, device=None):
    num_examples = (len(corpus_indices) - 1) // num_steps  # 总样本数
    epoch_size = num_examples // batch_size  # 小批量读取次数
    example_indices = list(range(num_examples))
    random.shuffle(example_indices)  # 打乱读取顺序

    def _data(pos):
        return corpus_indices[pos:pos + num_steps]  # 按下标返回一个样本

    for i in range(epoch_size):
        i *= batch_size  # 已读取的样本数
        batch_indices = example_indices[i: i + batch_size]  # 一个小批量中的所有样本下标
        X = [_data(j * num_steps) for j in batch_indices]  # 按下标生成对应的样本X
        Y = [_data(j * num_steps + 1) for j in batch_indices]  # 按下标生成对应的样本Y
        if device is not None:
            yield torch.FloatTensor(X).to(device), torch.FloatTensor(Y).to(device)
        else:
            yield torch.FloatTensor(X), torch.FloatTensor(Y)


def data_iter_consecutive(corpus_indices, batch_size, num_steps, device=None):
    if device is not None:
        corpus_indices = torch.FloatTensor(corpus_indices).to(device)
    else:
        corpus_indices = torch.FloatTensor(corpus_indices)
    data_len = len(corpus_indices)  # 总长度
    batch_len = data_len // batch_size
    # 将data变成矩阵
    indices = corpus_indices[0: batch_size * batch_len].view(batch_size, batch_len)
    epoch_size = (batch_len - 1) // num_steps  # 减1是为了防止取Y的时候导致下标越界
    for i in range(epoch_size):
        i *= num_steps
        X = indices[:, i: i + num_steps]
        Y = indices[:, i + 1: i + num_steps + 1]
        yield X, Y


def one_hot(label: torch.LongTensor, vocab_size):
    return torch.zeros(len(label), vocab_size, device=label.device).scatter_(1, label.view(-1, 1), 1)


def to_onehot(X, size):
    return [one_hot(x, size) for x in X.t()]


def one_hot2(index: torch.LongTensor, size):
    assert len(index.shape) == 2, "index must be 2-D"
    # shape: num_step, batch_size, vocab_size
    index = index.view(index.shape[0], index.shape[1], 1)
    return torch.zeros(index.shape[0], index.shape[1], size, device=index.device).scatter_(2, index, 1)


def predict_rnn(prefix, num_chars, rnn, params, init_rnn_state,
                num_hiddens, vocab_size, device, idx_to_char, char_to_idx):
    state = init_rnn_state(1, num_hiddens, device)  # 将隐藏状态初始化为 1×hidden
    output = [char_to_idx[prefix[0]]]
    for t in range(num_chars + len(prefix) - 1):  # prefix里的内容虽然不需要预测，但是需要放进rnn计算相应的隐藏状态
        X = to_onehot(torch.tensor([[output[-1]]], device=device), vocab_size)
        Y, state = rnn(X, state, params)
        if t < len(prefix) - 1:  # prefix的内容，只需要把原内容加进输出里即可
            output.append(char_to_idx[prefix[t + 1]])
        else:
            output.append(int(Y[0].argmax(dim=1).item()))  # 选取概率最大的字放入输出中
    return ''.join([idx_to_char[i] for i in output])


def grad_clipping(params, theta, device):
    norm = torch.FloatTensor([0]).to(device)
    for param in params:
        norm += (param.grad ** 2).sum()
    norm = norm.sqrt().item()
    if norm > theta:
        for param in params:
            param.grad *= theta / norm


def train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens,
                          vocab_size, device, corpus_indices, idx_to_char,
                          char_to_idx, is_random_iter, num_epochs, num_steps,
                          lr, clipping_theta, batch_size, pred_period,
                          pred_len, prefixes):
    # 选择读取数据的方式
    if is_random_iter:
        data_iter_fn = data_iter_random
    else:
        data_iter_fn = data_iter_consecutive
    params = get_params(vocab_size, num_hiddens, vocab_size)
    loss = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        # 若是相邻采样，则刚开始就初始化隐藏状态
        if not is_random_iter:
            state = init_rnn_state(batch_size, num_hiddens, device)
        l_sum, n, start = 0.0, 0, time()
        data_iter = data_iter_fn(corpus_indices, batch_size, num_steps, device)
        for X, Y in data_iter:
            if is_random_iter:  # 随机采样需要在每个批量之间初始化隐藏状态
                state = init_rnn_state(batch_size, num_hiddens, device)
            else:  # 相邻采样需要将隐藏状态从计算图中分离， 否则会因为图被释放掉无法追踪而报错
                for s in state:
                    s.detach_()

            X = X.type(torch.long)
            Y = Y.type(torch.long)
            inputs = to_onehot(X, vocab_size)  # 转化为one-hot表示法
            outputs, state = rnn(inputs, state, params)  # 计算输出值（index）以及隐藏状态
            outputs = torch.cat(outputs, dim=0)  # 每一行样本都有一个output来组成一个列表，此处将列表拼接起来
            y = Y.t().reshape(-1,)
            l = loss(outputs, y)

            l.backward()
            with torch.no_grad():
                grad_clipping(params, clipping_theta, device)  # 防止梯度爆炸
                sgd(params, lr)
            l_sum += l.item() * y.size(0)
            n += y.size(0)

        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (
                epoch + 1, math.exp(l_sum / n), time() - start))
            for prefix in prefixes:
                print(' -', predict_rnn(
                    prefix, pred_len, rnn, params, init_rnn_state,
                    num_hiddens, vocab_size, device, idx_to_char, char_to_idx))


def predict_rnn_nn(prefix, num_chars, model, device, idx_to_char, char_to_idx):
    state = model.begin_state(batch_size=1, device=device)
    output = [char_to_idx[prefix[0]]]
    for t in range(num_chars + len(prefix) - 1):
        X = torch.tensor([output[-1]], dtype=torch.long, device=device).view(1, 1)
        Y, state = model(X, state)
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
        else:
            output.append(int(Y.argmax(dim=1).item()))
    return ''.join([idx_to_char[i] for i in output])


def train_and_predict_rnn_nn(model, device,corpus_indices, idx_to_char,
                             char_to_idx, num_epochs, num_steps, lr, clipping_theta,
                             batch_size, pred_period, pred_len, prefixes):
    loss = nn.CrossEntropyLoss()
    trainer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0, weight_decay=0)

    for epoch in range(num_epochs):
        l_sum, n, start = 0.0, 0, time()
        data_iter = data_iter_consecutive(corpus_indices, batch_size, num_steps, device)
        state = model.begin_state(batch_size=batch_size, device=device)
        for X, Y in data_iter:
            if isinstance(state, tuple):
                for s in state:
                    s.detach_()
            else:
                state.detach_()

            output, state = model(X.type(torch.long), state)
            y = Y.t().reshape(-1,).type(torch.long)
            l = loss(output, y)
            l.backward()
            # params = [p for p in model.parameters()]
            # d2l.grad_clipping(params, clipping_theta, device)
            nn.utils.clip_grad_norm_(model.parameters(), clipping_theta)
            trainer.step()
            trainer.zero_grad()

            l_sum += l.item() * y.size(0)
            n += y.size(0)

        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (
                epoch + 1, math.exp(l_sum / n), time() - start))
            for prefix in prefixes:
                print(' -', predict_rnn_nn(prefix, pred_len, model, device, idx_to_char, char_to_idx))


class RNNModel(nn.Module):
    def __init__(self, rnn_layer, num_hiddens, vocab_size):
        super().__init__()
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = num_hiddens
        self.linear = nn.Linear(self.num_hiddens, vocab_size)

    def begin_state(self, batch_size, device=torch.device('cpu'), num_state=1):
        return torch.rand(num_state, batch_size, self.num_hiddens, device=device)

    def forward(self, inputs, state):
        X = one_hot2(inputs.t(), self.vocab_size)
        Y, state = self.rnn(X, state)
        output = self.linear(Y.view(-1, Y.shape[-1]))
        return output, state
