from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np

if __name__ == '__main__':
    writer = SummaryWriter('runs')
    img_path = './data/hymenoptera_data_label/train/ants_image/0013035.jpg'
    img_PIL = Image.open(img_path)
    img_array = np.array(img_PIL)

    writer.add_image('pic1', img_array, 1, dataformats='HWC')
    print(img_array.shape)

    for i in range(100):
        writer.add_scalar('y = x', i, i)

    writer.close()