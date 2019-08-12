import pickle
from pathlib import Path

import torch
import torchvision

if __name__ == '__main__':
    train = torchvision.datasets.FashionMNIST(root=Path('../data/training.pt'), train=True,
                                              download=True, transform=torchvision.transforms.ToTensor())
    test = torchvision.datasets.FashionMNIST(root=Path('../data/test.pt'), train=False,
                                             download=True, transform=torchvision.transforms.ToTensor())
    train_iter = torch.utils.data.DataLoader(train, batch_size=256, shuffle=True, num_workers=4)
    test_iter = torch.utils.data.DataLoader(test, batch_size=256, shuffle=True, num_workers=4)
    with open(Path('../data/torch_FashionMNIST_test'), 'wb') as f:
        pickle.dump(test_iter, f)
    with open(Path('../data/torch_FashionMNIST_train'), 'wb') as f:
        pickle.dump(train_iter, f)
