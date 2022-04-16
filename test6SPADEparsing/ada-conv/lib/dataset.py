import random
import warnings
from pathlib import Path
from torch.nn import functional as F
import matplotlib.pyplot as plt
import pandas as pd
from utils import generate_label
from PIL import Image
from torch.utils.data import IterableDataset, Dataset
from torchvision.transforms import ToTensor, Compose, Resize, CenterCrop, Normalize, RandomCrop
from torchvision.utils import save_image
import numpy as np
import torch
from PIL import ImageDraw
import json
import torchvision.transforms as transforms
import math


def files_in(dir):
    return list(sorted(Path(dir).glob('*')))


def save(img_tensor, file):
    if img_tensor.ndim == 4:
        assert len(img_tensor) == 1

    save_image(img_tensor, str(file))


def load(file):
    img = Image.open(str(file))
    img = img.convert('RGB')
    return img


def load_label(file):
    img = Image.open(str(file))
    return img


def style_transforms(size=256):
    # Style images must be 256x256 for AdaConv
    return Compose([
        Resize(size=size),  # Resize to keep aspect ratio
        CenterCrop(size=(size, size)),  # Center crop to square
        ToTensor()])


def content_transforms(min_size=None):
    # min_size is optional as content images have no size restrictions
    transforms = []
    if min_size:
        transforms.append(Resize(size=min_size))
    transforms.append(ToTensor())
    # transforms.append(Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    return Compose(transforms)


class StylizationDataset(Dataset):
    def __init__(self, content_files, style_files, content_transform=None, style_transform=None,
                 ):
        self.content_files = content_files
        self.style_files = style_files

        id = lambda x: x
        self.content_transform = id if content_transform is None else content_transform
        self.style_transform = id if style_transform is None else style_transform
        self.point_num = 18
        self.height = 256
        self.width = 256
        self.radius = 4
        self.transform = transforms.Compose([transforms.ToTensor()
                                                , transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.mean = (1, 1, 1)
        self.sl = 0.025
        self.sh = 0.05
        self.r1 = 0.25

    def __getitem__(self, idx):
        content_file, style_file = self.files_at_index(idx)
        content_file = str(style_file).replace('.jpg', '.png').replace('train2', 'trainp').replace('test22', 'testp')
        pose_file = str(style_file).replace('.jpg', '.json').replace('train2', 'pose').replace('test22', 'pose')

        content_img = load(content_file)
        parsing = load_label(content_file)
        style_img = load(style_file)

        content_img = self.content_transform(content_img)
        style_img = self.style_transform(style_img)
        # parsing = np.array(parsing).astype(np.long)
        # parsing = torch.from_numpy(parsing)
        # label_tensor = torch.zeros(20, 256, 256)
        # for i in range(20):
        #     label_tensor[i] += (parsing == i).float()

        # pose
        with open(pose_file) as f:
            pose_data = json.load(f)
        pose_maps = torch.zeros((self.point_num, self.height, self.width))
        im_pose = Image.new('RGB', (self.width, self.height))
        pose_draw = ImageDraw.Draw(im_pose)

        for i in range(self.point_num):
            one_map = Image.new('RGB', (self.width, self.height))
            draw = ImageDraw.Draw(one_map)
            if '%d' % i in pose_data:
                pointX = pose_data['%d' % i][0]
                pointY = pose_data['%d' % i][1]
                if pointX > 1 or pointY > 1:
                    draw.ellipse((pointX - self.radius, pointY - self.radius, pointX + self.radius,
                                  pointY + self.radius), 'white', 'white')
                    pose_draw.ellipse((pointX - self.radius, pointY - self.radius, pointX + self.radius,
                                       pointY + self.radius), 'white', 'white')
            one_map = self.transform(one_map)[0]
            pose_maps[i] = one_map

        # parsing
        parsing = np.array(parsing).astype(np.long)
        # print('parsing0',parsing0.shape)  256,256
        parsing = torch.from_numpy(parsing)

        # parsing_gt 20
        parsing_gt = torch.zeros(20, 256, 256)
        for i in range(20):
            parsing_gt[i] += (parsing == i).float()

        parsing_a = parsing.unsqueeze(0)
        # print('parsing', parsing_a.shape)  torch.Size([1, 256, 256])
        # 5通道parsing
        parsing_im_b_20 = torch.zeros(5, 256, 256)
        # hair hat
        # parsing_im_b_20[0] += (parsing == 2).float() + (parsing == 1).float()
        # # face、sunglasses、scarf、r_arm、l_arm、r_leg、l_leg、pants、r_shoe、l_shoe
        # parsing_im_b_20[1] += (parsing == 13).float() + (parsing == 3).float() + (parsing == 10).float() + \
        #                       (parsing == 14).float() + (parsing == 15).float() + (parsing == 16).float() + \
        #                       (parsing == 17).float() + (parsing == 8).float() + (parsing == 18).float() + (
        #                               parsing == 19).float()
        # # upper-clother coat dress
        # parsing_im_b_20[2] += (parsing == 5).float() + (parsing == 7).float() + (parsing == 6).float()
        # # pants skirt
        # parsing_im_b_20[3] += (parsing == 9).float() + (parsing == 12).float()
        # # background
        # parsing_im_b_20[4] += (parsing == 0).float() + (parsing == 4).float() + \
        #                       (parsing == 11).float()
        label_tensor = parsing_im_b_20  # torch.Size([5, 256, 256])

        # parsing_a mask  based on poses' key points
        with open(pose_file) as f:
            pose_data = json.load(f)
            # pose_data_keys = list(pose_data.keys())
            # #            print(pose_data_keys)# c ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17']
            # pose_data_values = list(pose_data.values())
        # 主要部位  pose_data_keys[1] /2，3，4，5，6，7，8，9，11，12
        area = parsing_gt.size()[1] * parsing_gt.size()[2]  # 整体区域 img.size()=tensor.Size([1,256,256])  c x h x w
        target_area = random.uniform(self.sl, self.sh) * area  # 黑盒 随机要抠掉的大小 w x h
        aspect_ratio = random.uniform(self.r1, 1 / self.r1)  # 随机要抠掉的长宽比 h / w
        h = int(round(math.sqrt(target_area * aspect_ratio)))  # 黑盒的h
        w = int(round(math.sqrt(target_area / aspect_ratio)))  # 黑盒的w
        # 随机取遮挡的pose部位
        random_point_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12]
        random_res = []
        for i in list(random_point_list):
            if '%d' % i in pose_data:
                random_res += '%d' % i

        random_point = int(random.choice(random_res))
        y = pose_data['%d' % random_point][1]  # w 对应pose坐标w
        x = pose_data['%d' % random_point][0]  # h 对应pose坐标h
        for i in range(20):
            parsing_gt[i, x:x + h, y:y + w] = self.mean[0]

        # parsing_a shift  based on poses' key points
        x_shift = random.uniform(-0.5, 0.5)
        y_shift = random.uniform(-0.5, 0.5)
        theta = torch.tensor([
            [1, 0, x_shift],
            [0, 1, y_shift]
        ], dtype=torch.float)
        grid = F.affine_grid(theta.unsqueeze(0), parsing_gt.unsqueeze(0).size())
        affine_parsing = F.grid_sample(parsing_gt.unsqueeze(0).float(), grid, )
        affine_parsing = affine_parsing[0]

        # mask
        mask = torch.zeros(1, 256, 256)
        # upper-clother coat dress
        mask[0] += (parsing == 5).float() + (parsing == 7).float() + (parsing == 6).float()
        # pants skirt
        mask[0] += (parsing == 9).float() + (parsing == 12).float()

        # label
        onehot_batch = []
        for onechannel in parsing:
            onehot_batch.append(torch.stack([onechannel == i for i in range(20)], dim=1).float())
            a = torch.stack(onehot_batch).squeeze()
        label = a.permute(2, 0, 1)

        # label_dim = torch.zeros(20, 1, 256, 256)
        # for i in range(256):
        #     for j in range(256):
        #         label_dim[parsing[i][j]][0] = 1
        # label = label_dim.permute(1, 0, 2, 3)[0]

        # parsing = np.array(parsing).astype(np.long)
        # parsing = torch.from_numpy(parsing)
        # parsing_im_b_20 = torch.zeros(1, 256, 256)
        # parsing_im_b_20[0] += (parsing == 5).float() + (parsing == 7).float() + (parsing == 6).float()
        # parsing_im_b_20[0] += (parsing == 9).float() + (parsing == 12).float()

        return {
            'label': label,  # torch.Size([20, 256, 256])
            'content': label_tensor,  # torch.Size([5, 256, 256])
            'style': style_img,  # torch.Size([3, 256, 256])
            'parsing': label_tensor,  # torch.Size([5, 256, 256])
            'mask': mask,  # torch.Size([1, 256, 256])
            'pose': pose_maps,  # torch.Size([18, 256, 256])
            'affine_parsing': affine_parsing,  # torch.Size([1, 256, 256])
            'parsing_gt': parsing_gt
        }

    def __len__(self):
        return len(self.content_files) * len(self.style_files)

    def files_at_index(self, idx):
        content_idx = idx % len(self.content_files)
        style_idx = idx // len(self.content_files)

        assert 0 <= content_idx < len(self.content_files)
        assert 0 <= style_idx < len(self.style_files)
        return self.content_files[content_idx], self.style_files[style_idx]
    #
    # def affine_parsing(self,size=256):
    #     return RandomCrop(
    #         size=(size,size),  # 裁剪后图片的尺寸
    #         padding=(50, 50, 0, 0),  # (左右填充多少，上下填充多少) （r/d/l/u）
    #         pad_if_needed=False,  # 当 size 大于原始图片的尺寸时，必须将该参数设置为 True，否则会报错
    #         fill=(255, 255, 255),  # 同 Pad 对象的参数
    #         padding_mode='constant'  # 同 Pad 对象的参数
    #     )


class EndlessDataset(IterableDataset):
    """
    Wrapper for StylizationDataset which loops infinitely.
    Usefull when training based on iterations instead of epochs
    """

    def __init__(self, *args, **kwargs):
        self.dataset = StylizationDataset(*args, **kwargs)

    def __iter__(self):
        while True:
            idx = random.randrange(len(self.dataset))

            try:
                yield self.dataset[idx]
            except Exception as e:
                files = self.dataset.files_at_index(idx)
                warnings.warn(f'\n{str(e)}\n\tFiles: [{str(files[0])}, {str(files[1])}]')
