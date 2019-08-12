from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def draw_loss(df):
    sns.relplot(x="epoch", y="loss", hue="ls_kind", kind="line", data=df)
    plt.show()


def data_load():
    all_features = pd.read_pickle(Path("../data/kaggle_features.pickle"))
    all_labels = pd.read_pickle(Path("../data/kaggle_labels.pickle"))
    return all_features, all_labels


def log_rmse(net, features, labels):
    y = net(features)
    clipped_preds = torch.max(y, torch.ones_like(y))
    rmse = torch.sqrt((clipped_preds.log() - labels.log()).pow(2).mean())
    return rmse.item()


def weight_init(net):
    for m in net.children():
        nn.init.normal_(m.weight)
        nn.init.zeros_(m.bias)


loss = nn.MSELoss()


def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    loss_df = pd.DataFrame(columns=("ls_kind", "epoch", "loss"))
    train_iter = DataLoader(TensorDataset(
        train_features, train_labels), batch_size, shuffle=True)
    trainer = torch.optim.Adam(net.parameters(),
                               learning_rate, weight_decay=weight_decay)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            trainer.step()
        # train_ls.append(log_rmse(net, train_features, train_labels))
        loss_df = loss_df.append({'ls_kind': "train_ls", 'epoch': epoch,
                                  'loss': log_rmse(net, train_features, train_labels)}, ignore_index=True)
        if test_labels is not None:
            loss_df = loss_df.append({'ls_kind': "test_ls", 'epoch': epoch,
                                      'loss': log_rmse(net, test_features, test_labels)}, ignore_index=True)
            # test_ls.append(log_rmse(net, test_features, test_labels))
    # return train_ls, test_ls, y_preds
    return loss_df


# 将第 i part设为测试集， 剩余的设为训练集
def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], dim=0)
            y_train = torch.cat([y_train, y_part], dim=0)
    return X_train, y_train, X_valid, y_valid


def k_fold(net, k, X_train, y_train, num_epochs,
           learning_rate, weight_decay, batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        loss_df = train(net, *data, num_epochs, learning_rate,
                        weight_decay, batch_size)
        last_train_ls = loss_df[loss_df['ls_kind'] == 'train_ls']['loss'].iloc[-1]
        last_test_ls = loss_df[loss_df['ls_kind'] == 'test_ls']['loss'].iloc[-1]
        train_l_sum += last_train_ls
        valid_l_sum += last_test_ls
        if i == 0:
            draw_loss(loss_df)
        print('fold %d, train rmse %f, valid rmse %f'
              % (i, last_train_ls, last_test_ls))
    return train_l_sum / k, valid_l_sum / k


if __name__ == '__main__':

    # 数据加载
    all_features, all_labels = data_load()
    train_labels = torch.tensor(all_labels).reshape(-1, 1).type(torch.float)
    n_train = train_labels.shape[0]
    train_features = torch.tensor(all_features[:n_train].values).type(torch.float)
    predict_features = torch.tensor(all_features[n_train:].values).type(torch.float)

    # 模型
    net = nn.Sequential(nn.Linear(train_features.shape[1], 1, True))
    weight_init(net)

    k, num_epochs, lr, weight_decay, batch_size = 5, 200, 5, 0.05, 32
    train_l, valid_l = k_fold(net, k, train_features, train_labels, num_epochs,
                                 lr, weight_decay, batch_size)
    print('%d-fold validation: avg train rmse %f, avg valid rmse %f'
          % (k, train_l, valid_l))



