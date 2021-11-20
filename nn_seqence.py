import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, padding=2)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 32, 5, padding=2)
        self.maxpool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(32, 64, 5, padding=2)
        self.maxpool3 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(1024, 64)
        self.linear2 = nn.Linear(64, 10)

        self.model1 = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        # x = self.conv1(x)
        # x = self.maxpool1(x)
        # x = self.conv2(x)
        # x = self.maxpool2(x)
        # x = self.conv3(x)
        # x = self.maxpool3(x)
        # x = self.flatten(x)
        # x = self.linear1(x)
        # x = self.linear2(x)
        x = self.model1(x)
        return x


def testNet():
    net = Net()
    writer = SummaryWriter("runs")
    input = torch.ones((64, 3, 32, 32))
    output = net(input)
    print(output.shape)
    writer.add_graph(net, input)


#def saveModel():



if __name__ == '__main__':
    net = Net()
    testNet()
    dataset = torchvision.datasets.CIFAR10("data/", transform= torchvision.transforms.ToTensor())
    dataloader = DataLoader(dataset, batch_size=64)
    epoch = 1
    if not os.path.isdir("./model"):
        os.mkdir("./model")
    torch.save(net, './model/net.pth')
    #saveModel()

    checkpoint = {
        "net": net.state_dict(),
        'optimizer': net.state_dict(),
        "epoch": epoch
    }

    torch.save(checkpoint, './model/net_last_%s.pth' % (str(epoch)))

    print(net)
