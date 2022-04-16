import argparse
from argparse import ArgumentParser

import torch

from lib import dataset
from lib.lightning.lightningmodel import LightningModel


def stylize_image(model, parsing_im_b_20, style, mask, pose_maps,label, content_size=None):
    device = next(model.parameters()).device

    # content = dataset.load(content_file)
    # style = dataset.load(style_file)
    #
    # content = dataset.content_transforms(content_size)(content)
    # style = dataset.style_transforms()(style)

    parsing_im_b_20 = parsing_im_b_20.to(device).unsqueeze(0)
    style = style.to(device).unsqueeze(0)
    label = label.to(device).unsqueeze(0)
    print('label',label.shape)
    mask = mask.to(device).unsqueeze(0)
    pose_maps = pose_maps.to(device)

    output = model(parsing_im_b_20, style, mask, pose_maps,label, return_embeddings=False)
    print('-----------------------------output', output.shape)
    return output[0].detach().cpu()


def parse_args():
    parser = ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--content', type=str, default='./content.png')
    parser.add_argument('--style', type=str, default='./style.png')
    parser.add_argument('--output', type=str, default='./output.png')
    parser.add_argument('--model', type=str, default='./model.ckpt')

    return vars(parser.parse_args())


if __name__ == '__main__':
    args = parse_args()

    model = LightningModel.load_from_checkpoint(checkpoint_path=args['model'])
    model = model.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    model.eval()

    with torch.no_grad():
        output = stylize_image(model, args['content'], args['style'])
    dataset.save(output, args['output'])
