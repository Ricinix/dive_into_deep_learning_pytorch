import MyD2l as d2l
import torch
import torchvision
from torch import nn
from torchsummary import summary
from torch.utils.data import DataLoader


data_dir = 'E:/Programming/jupyter_workspace/deep learning/data'
train_imgs = torchvision.datasets.ImageFolder(data_dir + '/hotdog/train')
test_imgs = torchvision.datasets.ImageFolder(data_dir + '/hotdog/test')

normalize = torchvision.transforms.Normalize(
    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

train_augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224, (0.1, 1), (0.5, 2)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    torchvision.transforms.RandomVerticalFlip(),
    torchvision.transforms.ToTensor()
])

test_augs = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor()
])


pretrained_net = torchvision.models.resnet18(pretrained=True)

output = nn.Linear(pretrained_net.fc.in_features, 2)
nn.init.xavier_uniform_(output.weight)
nn.init.zeros_(output.bias)
pretrained_net.fc = output
summary(pretrained_net, (3, 224, 224), device='cpu')


def get_param_with_name(module, *m_name):
    '''
    :param module: 模型
    :param m_name: 想要提取出的参数所在层名
    :return: 参数，最后一个是其余参数
    '''
    filter_params = []
    for name, m in module.named_modules():
        if name in m_name:
            filter_params += list(map(id, m.parameters()))
            yield m.parameters()
    yield filter(lambda p: id(p) not in filter_params, module.parameters())


output, other = get_param_with_name(pretrained_net, 'fc')


def train_fine_tuning(net, learning_rate, batch_size=128, num_epochs=5):
    net = net.to('cuda')
    train_imgs.transform = train_augs
    train_iter = DataLoader(train_imgs, batch_size, shuffle=True)
    test_imgs.transform = test_augs
    test_iter = DataLoader(test_imgs, batch_size)
    trainer = torch.optim.SGD([{'params': other, 'lr': learning_rate},
                              {'params': output, 'lr': learning_rate * 10}],
                              lr=learning_rate, weight_decay=0.1)
    d2l.train_ch5(net, train_iter, test_iter, trainer, num_epochs, device='cuda')


train_fine_tuning(pretrained_net, 0.001)
