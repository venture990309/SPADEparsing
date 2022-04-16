from argparse import ArgumentParser
from math import sqrt
from statistics import mean
from torch.nn import functional as F
import random
from utils import generate_label
from skimage.color import label2rgb
import pytorch_lightning as pl
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torchvision.utils import make_grid

from lib.adaconv.adaconv_model import AdaConvModel
from lib.adaconv.networks import define_D, GANLoss
from lib.adain.adain_model import AdaINModel
from lib.loss import MomentMatchingStyleLoss, GramStyleLoss, CMDStyleLoss, MSEContentLoss, PerceptualLoss, StyleLoss
import torchvision.transforms as transforms

class LightningModel(pl.LightningModule):
    @staticmethod
    def add_argparse_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # Add params of other models
        parser = AdaConvModel.add_argparse_args(parser)
        parser = AdaINModel.add_argparse_args(parser)
        parser.add_argument('--model-type', type=str, default='adaconv', choices=['adain', 'adaconv'])

        # Losses
        # mm = Moment Matching, gram = Gram matrix based, cmd = Central Moment Discrepancy
        parser.add_argument('--style-loss', type=str, default='mm', choices=['mm', 'gram', 'cmd'])
        parser.add_argument('--style-weight', type=float, default=10.0)
        parser.add_argument('--content-loss', type=str, default='mse', choices=['mse'])
        parser.add_argument('--content-weight', type=float, default=1.0)

        # Optimizer
        parser.add_argument('--lr', type=float, default=0.0001)
        parser.add_argument('--lr-decay', type=float, default=0.00005)
        return parser

    def __init__(self,
                 model_type,
                 alpha,
                 style_size, style_channels, kernel_size,
                 style_loss, style_weight,
                 content_loss, content_weight,
                 lr, lr_decay,
                 **_):
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.lr_decay = lr_decay
        self.style_weight = style_weight
        self.content_weight = content_weight

        # Style loss  mm
        if style_loss == 'mm':
            self.style_loss = MomentMatchingStyleLoss()
        elif style_loss == 'gram':
            self.style_loss = GramStyleLoss()
        elif style_loss == 'cmd':
            self.style_loss = CMDStyleLoss()
        else:
            raise ValueError('style_loss')

        self.criterionGAN = GANLoss(use_lsgan=True, tensor=torch.cuda.FloatTensor)

        # Content loss mse
        if content_loss == 'mse':
            self.content_loss = MSEContentLoss()
        else:
            raise ValueError('content_loss')

        self.l1 = torch.nn.L1Loss()
        self.vgg_loss = PerceptualLoss()
        self.y_style = StyleLoss()

        # Model type adaconv
        if model_type == 'adain':
            self.model = AdaINModel(alpha)
        elif model_type == 'adaconv':
            # self.model = AdaConvModel(style_size, style_channels, kernel_size)
            self.model = AdaConvModel(style_size, style_channels, kernel_size)
        else:
            raise ValueError('model_type')
        # self.netD = define_D(input_nc=3)
        self.netD2 = define_D(input_nc=21, ndf=64, which_model_netD='resnet', n_layers_D=3, norm='instance',
                              use_sigmoid=False, init_type='normal', gpu_ids=[2])

        self.netD = define_D(input_nc=20, ndf=64, which_model_netD='resnet', n_layers_D=3, norm='instance',
                             use_sigmoid=False, init_type='normal', gpu_ids=[2])

    def forward(self, parsing, style, mask, pose, label,affine_parsing ,return_embeddings=False):
        return self.model(parsing, style, mask, pose, label, affine_parsing,return_embeddings)

    def training_step(self, batch, batch_idx, optimizer_idx):
        return self.shared_step(batch, 'train', optimizer_idx)

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, 'val')

    def shared_step(self, batch, step, optimizer_idx=None):
        content, style, parsing, label, mask, pose, affine_parsing,parsing_gt = batch['content'], batch['style'], batch['parsing'],\
                                                                    batch['label'],batch['mask'], batch['pose'],\
                                                                    batch['affine_parsing'],batch['parsing_gt']

        # train generator
        output = self.model(parsing, style, mask, pose, label,affine_parsing, return_embeddings=False)

        # content_loss, style_loss, vgg_loss = self.loss(embeddings)
        # l1_loss = self.l1(output, style)
        #
        # # gen_fake_feat = self.netD(output)
        # # gen_fake_loss = self.criterionGAN(gen_fake_feat, True)
        # #
        # # gen_fake_feat2 = self.netD2(torch.cat((output, parsing), dim=1))
        # # gen_fake_loss2 = self.criterionGAN(gen_fake_feat2, True)
        #
        # # Log metrics
        # self.log(rf'{step}/loss_style', style_loss.item(), prog_bar=step == 'train')
        # self.log(rf'{step}/loss_content', content_loss.item(), prog_bar=step == 'train')
        # # self.log(rf'{step}/loss_gan', gen_fake_loss.item(), prog_bar=step == 'train')
        # # self.log(rf'{step}/loss_gan2', gen_fake_loss2.item(), prog_bar=step == 'train')
        # self.log(rf'{step}/loss_l1', l1_loss.item(), prog_bar=step == 'train')
        # self.log(rf'{step}/loss_vgg', vgg_loss.item(), prog_bar=step == 'train')
        # # Return output only for validation step
        # return style_loss + content_loss

        def cross_entropy(fake_out, real_parsing):
            # 取log
            fake_out = fake_out + 0.00001
            log_output = torch.log(fake_out)
            loss = - torch.sum((real_parsing * log_output))
            loss = loss / (256 * 256)
            return loss

        if optimizer_idx == 0:
            # style_loss = self.style_loss(output, style) * 50
            # vgg_loss = self.content_loss(output, style)
            # content_loss, style_loss = self.loss(embeddings)
            # l1_loss = self.l1(output, style)
            # vgg_loss = self.vgg_loss(output, style)
            # ystyle_loss = self.y_style(output, style) * 150
            #
            # gen_fake_feat = self.netD(output)
            # gen_fake_loss = self.criterionGAN(gen_fake_feat, True)
            #
            # gen_fake_feat2 = self.netD2(torch.cat((output, pose), dim=1))
            # gen_fake_loss2 = self.criterionGAN(gen_fake_feat2, True)
            # # Log metrics
            # # self.log(rf'{step}/loss_style', style_loss.item(), prog_bar=step == 'train')
            # # self.log(rf'{step}/loss_content', content_loss.item(), prog_bar=step == 'train')
            # self.log(rf'{step}/loss_ystyle', ystyle_loss.item(), prog_bar=step == 'train')
            # self.log(rf'{step}/loss_gan', gen_fake_loss.item(), prog_bar=step == 'train')
            # self.log(rf'{step}/loss_gan2', gen_fake_loss2.item(), prog_bar=step == 'train')
            # self.log(rf'{step}/loss_l1', l1_loss.item(), prog_bar=step == 'train')
            # self.log(rf'{step}/loss_vgg', vgg_loss.item(), prog_bar=step == 'train')
            # # Return output only for validation step
            # return ystyle_loss + l1_loss + vgg_loss + gen_fake_loss + gen_fake_loss2
            ce_loss = cross_entropy(output, parsing_gt) * 30
            gen_fake_feat = self.netD(output)
            gen_fake_loss = self.criterionGAN(gen_fake_feat, True)
            # ce_loss = self.cross_entropy(output, label) * 30

            # Log metrics
            self.log(rf'{step}/loss_gan', gen_fake_loss.item(), prog_bar=step == 'train')
            self.log(rf'{step}/loss_ce', ce_loss.item(), prog_bar=step == 'train')
            # return gen_fake_loss + ce_loss
            return gen_fake_loss + ce_loss

        # train discriminator
        if optimizer_idx == 1:
            # dis_real_feat = self.netD(output)
            # dis_fake_feat = self.netD(output.detach())
            # dis_real_loss = self.criterionGAN(dis_real_feat, True)
            # dis_fake_loss = self.criterionGAN(dis_fake_feat, False)
            # # Log metrics
            # self.log(rf'{step}/loss_d_gan_real', dis_real_loss.item(), prog_bar=step == 'train')
            # self.log(rf'{step}/loss_d_gan_fake', dis_fake_loss.item(), prog_bar=step == 'train')
            # # Return output only for validation step
            # return dis_real_loss + dis_fake_loss
            dis_real_feat = self.netD(parsing_gt)
            dis_fake_feat = self.netD(output.detach())
            dis_real_loss = self.criterionGAN(dis_real_feat, True)
            dis_fake_loss = self.criterionGAN(dis_fake_feat, False)
            # Log metrics
            self.log(rf'{step}/loss_d_gan_real', dis_real_loss.item(), prog_bar=step == 'train')
            self.log(rf'{step}/loss_d_gan_fake', dis_fake_loss.item(), prog_bar=step == 'train')
            # Return output only for validation step
            return dis_real_loss + dis_fake_loss
        #
        # # train discriminator2
        # if optimizer_idx == 2:
        #     dis_real_feat2 = self.netD2(torch.cat((style, pose), dim=1))
        #     dis_fake_feat2 = self.netD2(torch.cat((output.detach(), pose), dim=1))
        #     dis_real_loss2 = self.criterionGAN(dis_real_feat2, True)
        #     dis_fake_loss2 = self.criterionGAN(dis_fake_feat2, False)
        #     # Log metrics
        #     self.log(rf'{step}/loss_d_gan_real2', dis_real_loss2.item(), prog_bar=step == 'train')
        #     self.log(rf'{step}/loss_d_gan_fake2', dis_fake_loss2.item(), prog_bar=step == 'train')
        #     # Return output only for validation step
        #     return dis_real_loss2 + dis_fake_loss2

        if step == 'val':
            #     output_cpu = output.cpu().float()
            output_img = generate_label(output, 256, 256)   #tensorlabel2img
#            print('output_img',output_img.shape)  output_img torch.Size([1, 3, 256, 256])

            return {
                'output_img': output_img,
            }

    def validation_epoch_end(self, outputs):

        if self.global_step == 0:
            return

        with torch.no_grad():
            imgs = [x['output_img'] for x in outputs]
            imgs = [img for triple in imgs for img in triple]
            nrow = int(sqrt(len(imgs)))
            grid = make_grid(imgs, nrow=nrow, padding=0)
            logger = self.logger.experiment
            logger.add_image(rf'val_img', grid, global_step=self.global_step + 1)

    # def loss(self, embeddings):
    #     # Content
    #     content_loss = self.content_loss(embeddings['style'][-1], embeddings['output'][-1])
    #     # style_loss = self.style_loss(embeddings['style'][-1], embeddings['output'][-1])
    #
    #     # Style
    #     style_loss = []
    #     for (style_features, output_features) in zip(embeddings['style'], embeddings['output']):
    #         style_loss.append(self.style_loss(style_features, output_features))
    #     style_loss = sum(style_loss)
    #
    #     # vgg_loss = []
    #     # for (style_features, output_features) in zip(embeddings['style'], embeddings['output']):
    #     #     vgg_loss.append(self.l1(style_features, output_features))
    #     # vgg_loss = sum(vgg_loss)
    #
    #     return content_loss * 0.01, style_loss * 0.1

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=self.lr)

        def lr_lambda(iter):
            return 1 / (1 + 0.0002 * iter)

        def lr_lambda2(iter):
            return 1 / 1

        def lr_lambda3(iter):
            return 1 / 1

        lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

        opt_d = torch.optim.Adam(self.netD.parameters(), lr=0.0002)
        lr_scheduler2 = LambdaLR(opt_d, lr_lambda=lr_lambda2)

        # opt_d2 = torch.optim.Adam(self.netD2.parameters(), lr=0.0002)
        # lr_scheduler3 = LambdaLR(opt_d2, lr_lambda=lr_lambda3)

        return (
            {'optimizer': optimizer,
             'lr_scheduler': {
                 "scheduler": lr_scheduler,
                 "interval": "step",
                 "frequency": 1,
             }},
            {'optimizer': opt_d,
             'lr_scheduler': {
                 "scheduler": lr_scheduler2,
                 "interval": "step",
                 "frequency": 1,
             }},
            # {'optimizer': opt_d2,
            # 'lr_scheduler': {
            # "scheduler": lr_scheduler3,
            # "interval": "step",
            # "frequency": 1,
            # }}
        )