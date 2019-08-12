import torch
from torch import nn
import MyD2l as d2l


num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 1000, 1000
drop_prob1, drop_prob2 = 0.2, 0.5
num_epochs, lr, batch_size = 5, 0.5, 256


net = nn.Sequential(
    nn.Linear(num_inputs, num_hiddens1),
    nn.ReLU(),
    nn.Dropout(drop_prob1),
    nn.Linear(num_hiddens1, num_hiddens2),
    nn.ReLU(),
    nn.Dropout(drop_prob2),
    nn.Linear(num_hiddens2, num_outputs)
)

if __name__ == '__main__':
    trainer = torch.optim.SGD(net.parameters(), lr, weight_decay=0.01)
    loss = nn.CrossEntropyLoss()
    train_iter, test_iter = d2l.data_load()
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer=trainer, input_num=num_inputs)
