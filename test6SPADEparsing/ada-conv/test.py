import argparse
from argparse import ArgumentParser
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from torchvision.utils import make_grid
from tqdm import tqdm

from lib import dataset
from lib.lightning.lightningmodel import LightningModel
from stylize import stylize_image
import pandas as pd
from PIL import Image
from PIL import ImageDraw
import numpy as np
import json
import torchvision.transforms as transforms
import os


def resize(img, size):
    c, h, w = img.size()
    if h < w:
        small_size = size[0]
    else:
        small_size = size[1]

    img = TF.resize(img, small_size)
    img = TF.center_crop(img, size)
    return img

def load(file):
    img = Image.open(str(file))
    img = img.convert('RGB')
    return img

def load_label(file):
    img = Image.open(str(file))
    return img


def parse_args():
    parser = ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--content-dir', type=str, default='/data/hj/Projects/new_task/Pose-Transfer/fashion_data/vala/testp')
    parser.add_argument('--style-dir', type=str, default='/data/hj/Projects/new_task/Pose-Transfer/fashion_data/test22')
    parser.add_argument('--output-dir', type=str, default='./test_images/output')
    parser.add_argument('--model', type=str, default='./model.ckpt')
    parser.add_argument('--save-as', type=str, default='png')
    parser.add_argument('--content-size', type=int, default=512,
                        help='Content images are resized such that the smaller edge has this size.')

    return vars(parser.parse_args())


if __name__ == '__main__':
    args = parse_args()
    ext = args['save_as']
    content_transform = dataset.content_transforms(args['content_size'])
    style_transform = dataset.style_transforms()

    point_num = 18
    height = 256
    width = 256
    radius = 4
    transform = transforms.Compose([transforms.ToTensor()
                                            , transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    content_files = dataset.files_in(args['content_dir'])
    style_files = dataset.files_in(args['style_dir'])
    output_dir = Path(args['output_dir'])
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    file_path = '/data/hj/Projects/new_task/Pose-Transfer/fashion_data/test'
    filelist = os.listdir(file_path)

    model = LightningModel.load_from_checkpoint(checkpoint_path=args['model'])
    model = model.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    model.eval()

    pairs_file = pd.read_csv('/data/hj/Projects/new_task/Pose-Transfer/fashion_data/fasion-resize-pairs-test.csv')
    size = len(pairs_file)
    pairs = []
    for i in range(size):
        if pairs_file.iloc[i]['from'] in filelist:
            pair = [pairs_file.iloc[i]['from'], pairs_file.iloc[i]['to']]
            pairs.append(pair)

    pbar = tqdm(total=len(content_files) * len(style_files))
    with torch.no_grad():
        # Add style images at top row
        imgs = [style_transform(dataset.load(f)) for f in style_files]

        for idx in range(size):
            # print('---------------------------content', content)
            # # Add content images at left column
            # imgs.append(content_transform(dataset.load(content)))

            y_img = '/data/hj/Projects/new_task/Pose-Transfer/fashion_data/test/' + pairs[idx][0]
            y_parsing = y_img.replace('.jpg', '.png').replace('test', 'testp')
            t_pose = '/data/hj/Projects/new_task/Pose-Transfer/fashion_data/pose/' + pairs[idx][1].replace('.jpg','.json')
            t_parsing = '/data/hj/Projects/new_task/Pose-Transfer/fashion_data/testp/' + pairs[idx][1].replace('.jpg', '.png')

            print('-------------------------------------y', pairs[idx][0])

            # content_img = load(content_file)
            parsing = load_label(y_parsing)
            parsing1 = load_label(t_parsing)
            style_img = load(y_img)

            # content_img = content_transform(content_img)
            style_img = style_transform(style_img)
            # parsing = np.array(parsing).astype(np.long)
            # parsing = torch.from_numpy(parsing)
            # label_tensor = torch.zeros(20, 256, 256)
            # for i in range(20):
            #     label_tensor[i] += (parsing == i).float()

            with open(t_pose) as f:
                pose_data = json.load(f)
            pose_maps = torch.zeros((point_num, height, width))
            im_pose = Image.new('RGB', (width, height))
            pose_draw = ImageDraw.Draw(im_pose)

            for i in range(point_num):
                one_map = Image.new('RGB', (width, height))
                draw = ImageDraw.Draw(one_map)
                if '%d' % i in pose_data:
                    pointX = pose_data['%d' % i][0]
                    pointY = pose_data['%d' % i][1]
                    if pointX > 1 or pointY > 1:
                        draw.ellipse((pointX - radius, pointY - radius, pointX + radius,
                                      pointY + radius), 'white', 'white')
                        pose_draw.ellipse((pointX - radius, pointY - radius, pointX + radius,
                                           pointY + radius), 'white', 'white')
                one_map = transform(one_map)[0]
                pose_maps[i] = one_map

            parsing = np.array(parsing).astype(np.long)
            parsing = torch.from_numpy(parsing)
            parsing_im_b_20 = torch.zeros(5, 256, 256)
            parsing_im_b_20[0] += (parsing == 2).float() + (parsing == 1).float()
            parsing_im_b_20[1] += (parsing == 13).float() + (parsing == 3).float() + (parsing == 10).float() + \
                                  (parsing == 14).float() + (parsing == 15).float() + (parsing == 16).float() + (
                                              parsing == 17).float() + (parsing == 8).float() + (
                                              parsing == 18).float() + (parsing == 19).float()
            parsing_im_b_20[2] += (parsing == 5).float() + (parsing == 7).float() + (parsing == 6).float()
            parsing_im_b_20[3] += (parsing == 9).float() + (parsing == 12).float()
            parsing_im_b_20[4] += (parsing == 0).float() + (parsing == 4).float() + \
                                  (parsing == 11).float()
            label_tensor_y = parsing_im_b_20

            parsing = np.array(parsing1).astype(np.long)
            parsing = torch.from_numpy(parsing)
            parsing_im_b_20 = torch.zeros(5, 256, 256)
            parsing_im_b_20[0] += (parsing == 2).float() + (parsing == 1).float()
            parsing_im_b_20[1] += (parsing == 13).float() + (parsing == 3).float() + (parsing == 10).float() + \
                                  (parsing == 14).float() + (parsing == 15).float() + (parsing == 16).float() + (
                                              parsing == 17).float() + (parsing == 8).float() + (
                                              parsing == 18).float() + (parsing == 19).float()
            parsing_im_b_20[2] += (parsing == 5).float() + (parsing == 7).float() + (parsing == 6).float()
            parsing_im_b_20[3] += (parsing == 9).float() + (parsing == 12).float()
            parsing_im_b_20[4] += (parsing == 0).float() + (parsing == 4).float() + \
                                  (parsing == 11).float()
            label_tensor_t = parsing_im_b_20

            mask = torch.zeros(1, 256, 256)
            mask[0] += (parsing == 5).float() + (parsing == 7).float() + (parsing == 6).float()
            mask[0] += (parsing == 9).float() + (parsing == 12).float()

            label = torch.zeros(5, 256, 256)
            for i in range(5):
                label[i] += parsing_im_b_20[i].float()
            device = next(model.parameters()).device

            # content = dataset.load(content_file)
            # style = dataset.load(style_file)
            #
            # content = dataset.content_transforms(content_size)(content)
            # style = dataset.style_transforms()(style)

            label_tensor_y = label_tensor_y.to(device).unsqueeze(0)
            label_tensor_t = label_tensor_t.to(device).unsqueeze(0)
            style = style_img.to(device).unsqueeze(0)
            mask = mask.to(device).unsqueeze(0)
            pose_maps = pose_maps.to(device).unsqueeze(0)

            output = model(label_tensor_y, style, label_tensor_t, pose_maps, label,False)
            print('--------------------------out', output.shape)

            # output = stylize_image(model, parsing_im_b_20, style_img, mask, pose_maps, content_size=args['content_size'])
            dataset.save(output[0], output_dir.joinpath(pairs[idx][0] + pairs[idx][1]))



            # for j, style in enumerate(style_files):
                # Stylize content-style pair



                # imgs.append(output)
                # pbar.update(1)

        # Make all same size for table
        # avg_h = int(sum([img.size(1) for img in imgs]) / len(imgs))
        # avg_w = int(sum([img.size(2) for img in imgs]) / len(imgs))
        # imgs = [resize(img, [avg_h, avg_w]) for img in imgs]
        # imgs = [torch.ones((3, avg_h, avg_w)), *imgs]  # Add empty top left square.
        # grid = make_grid(imgs, nrow=len(style_files) + 1, padding=16, pad_value=1)
        # dataset.save(grid, output_dir.joinpath(f'table.{ext}'))
