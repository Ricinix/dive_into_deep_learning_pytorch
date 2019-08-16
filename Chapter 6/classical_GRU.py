import MyD2l as d2l
import torch


def get_params(num_inputs, num_hiddens, num_outputs, device=torch.device('cuda')):
    def _one(*shape):
        return torch.normal(torch.zeros(*shape), 0.01).to(device)

    def _three():
        return (_one(num_inputs, num_hiddens),
                _one(num_hiddens, num_hiddens),
                torch.zeros(num_hiddens, device=device))

    W_xz, W_hz, b_z = _three()  # 更新门
    W_xr, W_hr, b_r = _three()  # 重置门
    W_xh, W_hh, b_h = _three()  # 候选隐藏状态参数

    # 输出层
    W_hq = _one(num_hiddens, num_outputs)
    b_q = torch.zeros(num_outputs, device=device)

    params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params


def init_gru_state(batch_size, num_hiddens, device=torch.device('cuda')):
    return torch.zeros(batch_size, num_hiddens, device=device),


def gru(inputs, state, params):
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H,  = state
    outputs = []
    for X in inputs:
        Z = torch.sigmoid(torch.mm(X, W_xz) + torch.mm(H, W_hz) + b_z)
        R = torch.sigmoid(torch.mm(X, W_xr) + torch.mm(H, W_hr) + b_r)
        H_tilda = torch.tanh(torch.mm(X, W_xh) + torch.mm(R * H, W_hh) + b_h)
        H = Z * H + (1 - Z) * H_tilda
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)
    return outputs, (H, )


if __name__ == '__main__':
    corpus_indices, char_to_idx, idx_to_char, vocab_size = d2l.load_data_jay_lyrics()
    num_epochs, num_steps, batch_size, lr, clipping_theta = 160, 35, 32, 1e2, 1e-2
    pred_period, pred_len, prefixes = 40, 50, ['分开', '不分开']
    num_hiddens, device = 256, torch.device('cuda')

    d2l.train_and_predict_rnn(gru, get_params, init_gru_state, num_hiddens, vocab_size,
                              device, corpus_indices, idx_to_char, char_to_idx,
                              False, num_epochs, num_steps, lr, clipping_theta,
                              batch_size, pred_period, pred_len, prefixes)
