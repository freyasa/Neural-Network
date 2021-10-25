from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import cv2


if __name__ == '__main__':
    img_path = './data/hymenoptera_data_label/train/ants_image/0013035.jpg'
    img = Image.open(img_path)

    writer = SummaryWriter('runs')

    tensor_trans = transforms.ToTensor()
    tensor_img = tensor_trans(img)
    cv_img = cv2.imread(img_path)

    writer.add_image("Tensor_img", tensor_img)
    writer.close()

