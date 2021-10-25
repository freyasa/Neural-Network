from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

if __name__ == '__main__':
    writer = SummaryWriter('runs')
    img = Image.open("./data/hymenoptera_data/train/ants/116570827_e9c126745d.jpg")
    print(img)

    # Transforms.ToTensor()
    trans_totensor = transforms.ToTensor()
    img_tensor = trans_totensor(img)
    writer.add_image("ToTensor", img_tensor)

    # transforms.Normalize() 其中的六个参数分别为三个layer的mean和std
    print(img_tensor[0][0][0])
    trans_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    img_norm = trans_norm(img_tensor)
    print(img_norm[0][0][0])
    writer.add_image("ToNormalize", img_norm, 0)

    trans_norm = transforms.Normalize([1, 3, 5], [0.5, 0.6, 0.7])
    img_norm = trans_norm(img_tensor)
    writer.add_image("ToNormalize", img_norm, 1)

    # transforms.Resize() 其中一个pair参数为resize后的图片大小
    print(img.size)
    trans_resize = transforms.Resize((200, 200))
    img_resize = trans_resize(img)  # 要PIL的image
    print(type(img_resize))
    img_resize = trans_totensor(img_resize)
    writer.add_image("Resize", img_resize, 0)

    # Transforms.Compose() 可以将参数中的多个操作一起执行
    trans_resize = transforms.Resize(100)
    trans_compose = transforms.Compose([trans_resize, trans_totensor])
    img_resize = trans_compose(img)
    print(img_resize.shape)
    # img_resize = trans_totensor(img_resize)
    writer.add_image("Resize", img_resize, 1)


    # Transforms.RandomCrop()
    trans_random = transforms.RandomCrop((30, 50))
    trans_compose = transforms.Compose([trans_random, trans_totensor])
    for i in range(10):
        img_crop = trans_compose(img)
        writer.add_image("RandomCrop", img_crop, i)


    writer.close()