import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn
from torch.utils.tensorboard import SummaryWriter


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, 1, 0)

    def forward(self, x):
        x = self.conv1(x)
        return x


if __name__ == '__main__':
    dataset = torchvision.datasets.CIFAR10("data/", download=False, train=False, transform=torchvision.transforms.ToTensor())

    dataloader = DataLoader(dataset, batch_size=64)

    net = Net()
    print(net)

    writer = SummaryWriter("runs")
    step = 0

    for data in dataloader:
        imgs, targets = data
        output = net(imgs)
        output = torch.reshape(output, [-1, 3, 30, 30])
        writer.add_images("Input", imgs, step)
        writer.add_images("Output", output, step)
        step += 1
        print(output.shape)

    writer.close()