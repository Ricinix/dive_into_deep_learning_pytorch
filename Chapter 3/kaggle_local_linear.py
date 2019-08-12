from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def draw_loss(df):
    sns.relplot(x="epoch", y="loss", kind="line", data=df)
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


def loss(net, X, y, X_predict):
    tao = 0.3
    diff = X - X_predict
    diff_squared = torch.sum(diff.pow(2), dim=1, keepdim=True)
    weighted = torch.exp(- diff_squared / 2 * tao ** 2)

    J = weighted * (net(X) - y).pow(2)
    return J.mean()


def train(net, train_features, train_labels, predict_features,
          num_epochs, learning_rate, weight_decay, batch_size):
    loss_df = pd.DataFrame(columns=("epoch", "loss"))
    train_iter = DataLoader(TensorDataset(
        train_features, train_labels), batch_size, shuffle=True)
    trainer = torch.optim.Adam(net.parameters(),
                               learning_rate, weight_decay=weight_decay)

    y_predicts = []
    for predict_X in predict_features:
        weight_init(net)
        for epoch in range(num_epochs):
            for X, y in train_iter:
                trainer.zero_grad()
                l = loss(net, X, y, predict_X)
                l.backward()
                trainer.step()
            loss_df = loss_df.append({'epoch': epoch, 'loss': log_rmse(net, train_features, train_labels)},
                                     ignore_index=True)
            # test_ls.append(log_rmse(net, test_features, test_labels))
        y_predicts.append(net(predict_X))
    # return train_ls, test_ls, y_preds
    return loss_df, y_predicts


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

    loss_df, y_predicts = train(net, train_features, train_labels, predict_features, num_epochs,
                                lr, weight_decay, batch_size)
    draw_loss(loss_df)

    print('train rmse %f' % loss_df['loss'].iloc[-1])

    test_data = pd.read_csv(Path("../data/kaggle_house_pred_test.csv"))
    submission = pd.concat([test_data['Id'], pd.Series(y_predicts)], axis=1)
    submission.to_csv(Path('../data/submission.csv'), index=False)
