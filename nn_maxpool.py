import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, x):
        x = self.maxpool1(x)
        return x


if __name__ == '__main__':
    input = torch.tensor([[1, 2, 0, 3, 1],
                          [0, 1, 2, 3, 1],
                          [1, 2, 1, 0, 0],
                          [5, 2, 3, 1, 1],
                          [2, 1, 0, 1, 1]], dtype=torch.float32)
    input = torch.reshape(input, [-1, 1, 5, 5])

    dataset = torchvision.datasets.CIFAR10("data/", train=False, transform=torchvision.transforms.ToTensor(), download=False)
    dataloader = DataLoader(dataset, batch_size=64)

    net = Net()
    writer = SummaryWriter("runs")
    output = net(input)
    step = 0
    print(output)

    for data in dataloader:
        imgs, targets = data
        writer.add_images("Input", imgs, step)
        output = net(imgs)
        writer.add_images("output", output, step)
        step += 1

    writer.close()

