from torch import nn
import torch


class Net(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        output = input + 1;
        return output


if __name__ == '__main__':
    net = Net()
    x = torch.tensor(1.0)
    output = net(x)
    print(output)
