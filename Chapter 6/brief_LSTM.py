import MyD2l as d2l
import torch
from torch import nn


class RNNModel(nn.Module):
    def __init__(self, rnn_layer, num_hiddens, vocab_size):
        super().__init__()
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = num_hiddens
        self.linear = nn.Linear(self.num_hiddens, vocab_size)

    def begin_state(self, batch_size, device=torch.device('cpu'), num_state=1):
        return (torch.rand(num_state, batch_size, self.num_hiddens, device=device),
                torch.rand(num_state, batch_size, self.num_hiddens, device=device))

    def forward(self, inputs, state):
        X = d2l.one_hot2(inputs.t(), self.vocab_size)
        Y, state = self.rnn(X, state)
        output = self.linear(Y.view(-1, Y.shape[-1]))
        return output, state


if __name__ == '__main__':
    corpus_indices, char_to_idx, idx_to_char, vocab_size = d2l.load_data_jay_lyrics()
    num_epochs, num_steps, batch_size, lr, clipping_theta = 160, 35, 32, 1e2, 1e-2
    pred_period, pred_len, prefixes = 40, 50, ['分开', '不分开']
    num_hiddens, device = 256, torch.device('cuda')

    gru_layer = nn.LSTM(vocab_size, num_hiddens)
    model = RNNModel(gru_layer, num_hiddens, vocab_size).to(device)
    d2l.train_and_predict_rnn_nn(model, device, corpus_indices, idx_to_char, char_to_idx,
                                 num_epochs, num_steps, lr, clipping_theta, batch_size,
                                 pred_period, pred_len, prefixes)