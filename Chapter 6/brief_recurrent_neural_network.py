import MyD2l as d2l
import torch
from torch import nn
import time
import math


def one_hot2(index: torch.LongTensor, size):
    assert len(index.shape) == 2, "index must be 2-D"
    # shape: num_step, batch_size, vocab_size
    index = index.view(index.shape[0], index.shape[1], 1)
    return torch.zeros(index.shape[0], index.shape[1], size, device=index.device).scatter_(2, index, 1)


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
        l_sum, n, start = 0.0, 0, time.time()
        data_iter = d2l.data_iter_consecutive(corpus_indices, batch_size, num_steps, device)
        state = model.begin_state(batch_size=batch_size, device=device)
        for X, Y in data_iter:
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
                epoch + 1, math.exp(l_sum / n), time.time() - start))
            for prefix in prefixes:
                print(' -', predict_rnn_nn(prefix, pred_len, model, device, idx_to_char, char_to_idx))


if __name__ == '__main__':
    corpus_indices, char_to_idx, idx_to_char, vocab_size = d2l.load_data_jay_lyrics()
    num_hiddens, batch_size, num_steps, device = 256, 32, 35, torch.device('cuda')
    rnn_layer = nn.RNN(vocab_size, num_hiddens)
    # X = torch.rand(num_steps, batch_size, vocab_size)
    # state = torch.rand(1, 2, 256)
    # output, state_new = rnn_layer(X, state)
    # print(output.shape)
    # print(state_new.shape)
    model = RNNModel(rnn_layer, num_hiddens, vocab_size).to(device)
    # output = predict_rnn_nn('分开', 10, model, device, idx_to_char, char_to_idx)

    num_epochs, lr, clipping_theta = 250, 1e2, 1e-2
    pred_period, pred_len, prefixes = 50, 50, ['分开', '不分开']
    train_and_predict_rnn_nn(model, device, corpus_indices, idx_to_char, char_to_idx,
                             num_epochs, num_steps, lr, clipping_theta, batch_size,
                             pred_period, pred_len, prefixes)
