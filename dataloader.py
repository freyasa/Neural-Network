import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
if __name__ == '__main__':
    test_data = torchvision.datasets.CIFAR10("./data", train=False, transform=transforms.ToTensor(), download=False)
    test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=False)

    writer = SummaryWriter('runs')


    img, target = test_data[0]
    print(img.shape)
    print(target)

    step = 0
    for data in test_loader:
        imgs, targets = data
        writer.add_images("CIFAR10 Image", imgs, step)
        # print(imgs.shape)
        # print(targets)
        step += 1

    writer.close()
