import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
if __name__ == '__main__':

    dataset_transform = transforms.Compose([transforms.ToTensor()])

    writer = SummaryWriter('runs')
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, transform=dataset_transform, download=False)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, transform=dataset_transform, download=False)

    trans_toTensor = transforms.ToTensor()
    print(test_set.classes)

    # img_toTensor, target = test_set[0]
    # print(img_toTensor, test_set.classes[target])

    for i in range(10):
        img_toTensor, target = test_set[i]
        writer.add_image('CIFAR10 TestSet', img_toTensor, i)

    writer.close()
