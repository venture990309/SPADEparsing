import os
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F


def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])

def labelcolormap(N):
    if N == 20:  # CelebAMask-HQ
        cmap = np.array([ (0,0,0)

                , (128,0,0), (255,0,0), (0,85,0), (170,0,51), (255,85,0), (0,0,85), (0,119,221), (85,85,0), (0,85,85), (85,51,0), (52,86,128), (0,128,0)

                , (0,0,255), (51,170,221), (0,255,255), (85,255,170), (170,255,85), (255,255,0), (255,170,0),], # head face body leg bg
                        dtype=np.uint8)
      
    else:
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i
            for j in range(20):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (20 - j))
                g = g ^ (np.uint8(str_id[-2]) << (20 - j))
                b = b ^ (np.uint8(str_id[-3]) << (20 - j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
    return cmap


class Colorize(object):
    def __init__(self, n=20):
        self.cmap = labelcolormap(n)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image[0]).cpu()
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]
#         color_image torch.Size([3, 256, 256])
        return color_image


def tensor2label(label_tensor, n_label, imtype=np.uint8):
    # if n_label == 0:
    #     return tensor2im(label_tensor, imtype)
    label_tensor = label_tensor.cpu().float()
    if label_tensor.size()[0] > 1:
        label_tensor = label_tensor.max(0, keepdim=True)[1]
    label_tensor = Colorize(n_label)(label_tensor)
    # label_numpy = np.transpose(label_tensor.numpy(), (1, 2, 0))
    label_numpy = label_tensor.numpy()
    label_numpy = label_numpy / 255.0

    return label_numpy


def generate_label(inputs,h,w):
    pred_batch = []
    for input in inputs:
        input = input.reshape(1, 20, h, w)
        pred = np.squeeze(input.data.max(1)[1].cpu().numpy(), axis=0)
        pred_batch.append(pred)

    pred_batch = np.array(pred_batch)
    pred_batch = torch.from_numpy(pred_batch)

    label_batch = []
    for p in pred_batch:
        p = p.view(1, h,w)
        label_batch.append(tensor2label(p, 20))

    label_batch = np.array(label_batch)
    label_batch = torch.from_numpy(label_batch)
#   label_batch torch.Size([1, 3, 256, 256])

    return label_batch