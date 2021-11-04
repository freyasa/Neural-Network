import torchvision
import torch
from torch.utils.data import DataLoader
from torch import nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(196608, 10)

    def forward(self, x):
        x = self.linear1(x)
        return x


if __name__ == '__main__':
    dataset = torchvision.datasets.CIFAR10("data/", train=False, transform=torchvision.transforms.ToTensor())

    dataloader = DataLoader(dataset, batch_size=64, drop_last=True)

    net = Net()

    for data in dataloader:
        imgs, targets = data
        print(imgs.shape)

        # output = torch.reshape(imgs, (1, 1, 1, -1))
        output = torch.flatten(imgs)
        print(output.shape)

        output = net(output)

        print(output.shape)
