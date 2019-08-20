import MyD2l as d2l
import torch


def get_params(num_inputs, num_hiddens, num_outputs, device=torch.device('cuda')):
    def _one(*shape):
        return torch.normal(torch.zeros(*shape), 0.01).to(device)

    def _three():
        return (_one(num_inputs, num_hiddens),
                _one(num_hiddens, num_hiddens),
                torch.zeros(num_hiddens, device=device))

    W_xi, W_hi, b_i = _three()  # 输入门
    W_xf, W_hf, b_f = _three()  # 遗忘门
    W_xo, W_ho, b_o = _three()  # 输出门
    W_xc, W_hc, b_c = _three()  # 候选记忆细胞
    # 输出层
    W_hq = _one(num_hiddens, num_outputs)
    b_q = torch.zeros(num_outputs, device=device)

    params = [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params


def init_lstm_state(batch_size, num_hiddens, device=torch.device('cuda')):
    return (torch.zeros(batch_size, num_hiddens, device=device),
            torch.zeros(batch_size, num_hiddens, device=device))


def lstm(inputs, state, params):
    W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q = params
    (H, C) = state
    outputs = []
    for X in inputs:
        I = torch.sigmoid(torch.mm(X, W_xi) + torch.mm(H, W_hi) + b_i)
        F = torch.sigmoid(torch.mm(X, W_xf) + torch.mm(H, W_hf) + b_f)
        O = torch.sigmoid(torch.mm(X, W_xo) + torch.mm(H, W_ho) + b_o)
        C_tilda = torch.tanh(torch.mm(X, W_xc) + torch.mm(H, W_hc) + b_c)
        C = F * C + I * C_tilda
        H = O * C.tanh()
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)
    return outputs, (H, C)


if __name__ == '__main__':
    corpus_indices, char_to_idx, idx_to_char, vocab_size = d2l.load_data_jay_lyrics()
    num_epochs, num_steps, batch_size, lr, clipping_theta = 160, 35, 32, 1e2, 1e-2
    pred_period, pred_len, prefixes = 40, 50, ['分开', '不分开']
    num_hiddens, device = 256, torch.device('cuda')

    d2l.train_and_predict_rnn(lstm, get_params, init_lstm_state, num_hiddens, vocab_size,
                              device, corpus_indices, idx_to_char, char_to_idx,
                              False, num_epochs, num_steps, lr, clipping_theta,
                              batch_size, pred_period, pred_len, prefixes)