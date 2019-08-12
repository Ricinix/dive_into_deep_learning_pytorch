import random

import torch


def linreg(X, w, b):
    return torch.mm(X, w) + b


def square_loss(y_hat, y):
    return torch.sum((y_hat - y.reshape(y_hat.shape)).pow(2) / 2)


def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = torch.tensor(indices[i: min(i + batch_size, num_examples)])
        yield features[j], labels[j]


if __name__ == '__main__':
    # 初始化参数
    num_inputs = 2
    num_examples = 1000
    true_w = [2, -3.4]
    true_b = 4.2

    # 随机生成数据集
    features = torch.randn((num_examples, num_inputs))
    labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
    labels += torch.normal(torch.zeros_like(labels), 0.01)
    # print(features[0], labels[0])

    batch_size = 10
    # for X, y in data_iter(batch_size, features, labels):
    #     print(X, y)
    #     break

    # 初始化权重
    w = torch.normal(mean=torch.ones((num_inputs, 1)), std=0.01)
    w.requires_grad_(True)
    b = torch.zeros((1,), requires_grad=True)

    lr = 0.03
    num_epochs = 3
    net = linreg
    loss = square_loss

    for epoch in range(num_epochs):

        print("epoch:%d" % epoch)
        for X, y in data_iter(batch_size, features, labels):
            l = loss(net(X, w, b), y)
            print(l)
            l.backward()
            print(w)
            with torch.no_grad():
                w -= (lr * w.grad) / batch_size
                b -= (lr * b.grad) / batch_size
                w.grad.zero_()
                b.grad.zero_()
        train_l = loss(net(features, w, b), labels)
        print('epoch %d, loss %f' % (epoch + 1, train_l.mean().detach().numpy()))
    print(true_w, w.detach())
    print(true_b, b.detach())

    # 画图
    # sns.scatterplot(features[:, 1].numpy(), labels.numpy())
    # plt.show()

