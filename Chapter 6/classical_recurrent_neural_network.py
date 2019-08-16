import torch
from torch import nn
import MyD2l as d2l
import time
import math


def one_hot(label: torch.LongTensor, vocab_size):
    return torch.zeros(len(label), vocab_size, device=label.device).scatter_(1, label.view(-1, 1), 1)


def to_onehot(X, size):
    return [one_hot(x, size) for x in X.t()]


def get_params(num_inputs, num_hiddens, num_outputs, device=torch.device('cuda')):
    def _one(*shape):
        return torch.normal(torch.zeros(*shape), 0.01).to(device)

    W_xh = _one(num_inputs, num_hiddens)
    W_hh = _one(num_hiddens, num_hiddens)
    b_h = torch.zeros(num_hiddens, device=device)

    W_hq = _one(num_hiddens, num_outputs)
    b_q = torch.zeros(num_outputs, device=device)

    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params


def init_rnn_state(batch_size, num_hiddens, device=torch.device('cuda')):
    return (torch.zeros(batch_size, num_hiddens, device=device), )


def rnn(inputs, state, params):
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)
    return outputs, (H, )


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
        data_iter_fn = d2l.data_iter_random
    else:
        data_iter_fn = d2l.data_iter_consecutive
    params = get_params(vocab_size, num_hiddens, vocab_size)
    loss = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        # 若是相邻采样，则刚开始就初始化隐藏状态
        if not is_random_iter:
            state = init_rnn_state(batch_size, num_hiddens, device)
        l_sum, n, start = 0.0, 0, time.time()
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
                d2l.sgd(params, lr)
            l_sum += l.item() * y.size(0)
            n += y.size(0)

        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (
                epoch + 1, math.exp(l_sum / n), time.time() - start))
            for prefix in prefixes:
                print(' -', predict_rnn(
                    prefix, pred_len, rnn, params, init_rnn_state,
                    num_hiddens, vocab_size, device, idx_to_char, char_to_idx))


if __name__ == '__main__':
    corpus_indices, char_to_idx, idx_to_char, vocab_size = d2l.load_data_jay_lyrics()
    # one_hot(torch.LongTensor([0, 2]), vocab_size)
    # print(one_hot)

    # X = torch.arange(10).view(2, 5).cuda()
    # print(X)
    # inputs = to_onehot(X, vocab_size)
    # print(len(inputs), inputs[0].shape)
    # for input in inputs:
    #     print(input)
    # params = get_params(vocab_size, 256, vocab_size)

    # state = init_rnn_state(X.shape[0], 256, torch.device('cuda'))
    # inputs = to_onehot(X, vocab_size)
    # outputs, state_new = rnn(inputs, state, params)
    # print(len(outputs), outputs[0].shape, state_new[0].shape)
    # o = predict_rnn('分开', 10, rnn, params, init_rnn_state, 256, vocab_size,
    #                 torch.device('cuda'), idx_to_char, char_to_idx)
    # print(o)

    num_hiddens, device = 256, torch.device('cuda')
    num_epochs, num_steps, batch_size, lr, clipping_theta = 250, 35, 32, 1e2, 1e-2
    pred_period, pred_len, prefixes = 50, 50, ['分开', '不分开']
    train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens,
                          vocab_size, device, corpus_indices, idx_to_char,
                          char_to_idx, False, num_epochs, num_steps, lr,
                          clipping_theta, batch_size, pred_period, pred_len,
                          prefixes)
