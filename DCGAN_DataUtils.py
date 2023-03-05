
from os import listdir
from os.path import join
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize

def is_image_file(filename):
    # 判断文件后缀
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

def calculate_valid_crop_size(crop_size, upscale_factor):
    # 有效裁剪片段
    return crop_size - (crop_size % upscale_factor)

def train_hr_transform(crop_size):
    # Compose 组合多个操作 下同
    return Compose([
        # 随机裁剪
        RandomCrop(crop_size),
        # 变为张量
        ToTensor(),
    ])

def train_lr_transform(crop_size, upscale_factor):  # 裁剪好的图像处理成低分辨率的图片
    return Compose([
        # 变为图片
        ToPILImage(),
        # 整除下采样
        Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC),
        ToTensor()
    ])

def display_transform():
    return Compose([
        ToPILImage(),
        Resize(400),
        CenterCrop(400),
        ToTensor()
    ])

# 对于数据进行预处理
class TrainDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, crop_size, upscale_factor):
        super(TrainDatasetFromFolder, self).__init__()
        # 获取图片
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        # 有效裁剪尺寸
        crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        # 随机裁剪原图像
        self.hr_transform = train_hr_transform(crop_size)
        # 将裁剪好的图像处理成低分辨率的图片
        self.lr_transform = train_lr_transform(crop_size, upscale_factor)

    def __getitem__(self, index):
        hr_image = self.hr_transform(Image.open(self.image_filenames[index]).convert('RGB'))     #
        lr_image = self.lr_transform(hr_image)
        return lr_image, hr_image

    def __len__(self):
        return len(self.image_filenames)