import MyD2l as d2l
import torch
from torch import nn


if __name__ == '__main__':
    corpus_indices, char_to_idx, idx_to_char, vocab_size = d2l.load_data_jay_lyrics()
    num_epochs, num_steps, batch_size, lr, clipping_theta = 160, 35, 32, 1e2, 1e-2
    pred_period, pred_len, prefixes = 40, 50, ['分开', '不分开']
    num_hiddens, device = 256, torch.device('cuda')

    gru_layer = nn.GRU(vocab_size, num_hiddens)
    model = d2l.RNNModel(gru_layer, num_hiddens, vocab_size).to(device)
    d2l.train_and_predict_rnn_nn(model, device, corpus_indices, idx_to_char, char_to_idx,
                                 num_epochs, num_steps, lr, clipping_theta, batch_size,
                                 pred_period, pred_len, prefixes)