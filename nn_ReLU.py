import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.relu1 = nn.ReLU()

    def forward(self, x):
        x = self.relu1(x)
        return x


if __name__ == '__main__':

    dataset = torchvision.datasets.CIFAR10("data", train=False, transform=torchvision.transforms.ToTensor())
    dataloader = DataLoader(dataset, batch_size=64)

    input = torch.tensor([[1, -0.5],
                          [-1, 3]])

    net = Net()
    input = torch.reshape(input, (-1, 1, 2, 2))

    step = 0
    writer = SummaryWriter("runs")

    for data in dataloader:
        imgs, targets = data
        output = net(imgs)

        writer.add_images("Input", imgs, step)
        writer.add_images("Output", output, step)
        step += 1

    writer.close()
