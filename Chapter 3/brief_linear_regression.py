import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class LinearRegression(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1, bias=True)

    def forward(self, X):
        y_hat = self.linear(X)
        return y_hat


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
    train_ds = TensorDataset(features, labels.view((-1, 1)))

    # 初始化模型
    module = LinearRegression()
    nn.init.normal_(module.linear.weight, 0, 0.01)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(module.parameters(), lr=0.03)
    for epoch in range(3):
        for X, y in DataLoader(dataset=train_ds, batch_size=10, shuffle=True):
            y_predict = module(X)
            optimizer.zero_grad()
            loss = criterion(y_predict, y)
            loss.backward()
            optimizer.step()
        y_predict = module(features)
        print(criterion(y_predict, labels.view(y_predict.shape)))
    y_predict = module(features).detach()
    print(true_w, true_b)
    print(module.linear.weight, module.linear.bias)
