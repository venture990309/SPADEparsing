from argparse import ArgumentParser

import torchinfo
from torch import nn
import torch
from PIL import Image
import os
from utils import generate_label

from lib.adaconv.adaconv import AdaConv2d
from lib.adaconv.kernel_predictor import KernelPredictor
from lib.adaconv.res_conv import res_conv
from lib.vgg import VGGEncoder
from lib.vgg import VGGDecoder
import torch.nn.utils.spectral_norm as spectral_norm
import torch.nn.functional as F
import torch
from torch.nn import init
import functools
import numpy as np
from skimage.color import label2rgb


class AdaConvModel(nn.Module):
    @staticmethod
    def add_argparse_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--style-size', type=int, default=256, help='Size of the input style image.')
        parser.add_argument('--style-channels', type=int, default=512,
                            help='Number of channels for the style descriptor.')
        parser.add_argument('--kernel-size', type=int, default=3, help='The size of the predicted kernels.')
        return parser

    def __init__(self, style_size, style_channels, kernel_size):
        super().__init__()
        # self.encoder = VGGEncoder()
        # self._decoder = ParsingDecoder(style_channels=style_channels, kernel_size=kernel_size)
        # style_in_shape = (
        #     self.encoder.out_channels, style_size // self.encoder.scale_factor, style_size // self.encoder.scale_factor)
        # style_out_shape = (style_channels, kernel_size, kernel_size)
        # self.style_encoder = GlobalStyleEncoder(in_shape=style_in_shape, out_shape=style_out_shape)
        # self.decoder = AdaConvDecoder(style_channels=style_channels, kernel_size=kernel_size)
        self.downscale = nn.Sequential(
            nn.Conv2d(38, 64, 3, 2, padding=1), # 18+20+20
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            #
            nn.Conv2d(64, 128, 3, 2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            #
            nn.Conv2d(128, 256, 3, 2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 512, 3, 1, padding=1),
        )

        self.spade0 = SPADEResnetBlock(512, 20,512)
        self.spade1 = SPADEResnetBlock(256, 20,256)
        self.spade2 = SPADEResnetBlock(128,20, 128)
        self.spade3 = SPADEResnetBlock(64, 20,64)



        self.upscale0 = nn.Sequential(
            ResBlocks(1, 512, 'none', 'relu'),
            nn.Conv2d(512, 256, 3, padding=1),

            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),)
        self.upscale1 = nn.Sequential(
            ResBlocks(1, 256, 'none', 'relu'),
            nn.Conv2d(256, 128, 3, padding=1),

            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
        )
        self.upscale2 = nn.Sequential(
            #
            ResBlocks(1, 128, 'none', 'relu'),
            nn.Conv2d(128, 64, 3, padding=1),

            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            #
        )
        self.upscale3 = nn.Sequential(
            ResBlocks(1, 64, 'none', 'relu'),
            nn.Conv2d(64, 20, 3, padding=1),
            nn.Softmax(dim=1),
        )

    def forward(self, content, style, mask, pose, label, affine_parsing,return_embeddings=False):
        # parsing = content
        # ------对parsing进行变换------

        # self.encoder.freeze()
        # pose_enc = self.enc_pose(pose)  # pose_encoder
        # style_embeddings, style_embeddingss = self._encode(style, parsing, mask)
        # glob_pose = self.pose_att(pose_enc,style_embeddings[-1])
        # print(glob_pose.shape) torch.Size([1, 512, 32, 32])
        # glob_pose2img = self.warp2img(glob_pose)
        # glob_posenew = glob_pose.reshape(1,8,256,256)
        # print('label',label.shape)
        content_embeddings = self.downscale(torch.cat((pose, label), dim=1)) # 1 x 512 x32 x32
        # content_embeddings = self.downscale(torch.cat((pose, label), dim=1))
        content0 = self.upscale0(self.spade0(content_embeddings,affine_parsing))

        content1 = self.upscale1(self.spade1(content0,affine_parsing))
        content2 = self.upscale2(self.spade2(content1,affine_parsing))
        output = self.upscale3(self.spade3(content2,affine_parsing))  # 1 x 20 x256 x 256
        #        print('content_embeddings',content_embeddings.shape) torch.Size([1, 512, 32, 32])

        # pose torch.Size([1, 18, 256, 256])
        # label torch.Size([1, 5, 256, 256])
        # output = self._decode(content_embeddings, style_embeddingss, parsing, parsing, style_embeddings)
        # output = self.upscale(content_embeddings)

        # print('output', output.shape)  label torch.Size([1, 5, 256, 256])
        if return_embeddings:
            output_embeddings = self.encoder(output)
            embeddings = {
                # 'content': style_embeddings,
                # 'style': style_embeddings,
                # 'output': output_embeddings
            }
            return output, embeddings
        else:
            return output

    # def _encode(self, style, parsing, mask):
    #     y_style = style[0].unsqueeze(0)
    #     # t_style = style[1].unsqueeze(0)
    #     style_embeddings = self.encoder(y_style)
    #     style_embeddingss = []
    #     for i in range(parsing.shape[0]):
    #         for j in range(parsing.shape[1]):
    #             style_embeddingss.append(self.encoder(y_style * parsing[i][j])[-1])
    #     return style_embeddings, style_embeddingss

    # def _decode(self, content_embedding, style_embedding, parsing, mask, style_attn):
    #     style_embedding = self.style_encoder(style_embedding, parsing)  # list s
    #     # torch.Size([1, 512, 1, 1]) torch.Size([1, 256, 1, 1]) torch.Size([1, 128, 1, 1]) torch.Size([1, 64, 1, 1])
    #     output = self.decoder(content_embedding, style_embedding, parsing, style_attn)
    #     return output
    def _decode(self, content_embedding):
        output = self._decoder(content_embedding)
        return output

class SPADEResnetBlock(nn.Module):
    def __init__(self, fin,fseg,fout):
        super().__init__()
        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fout, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fin, fout, kernel_size=3, padding=1)

        # define normalization layers
        self.norm_0 = SPADE(fin,fseg)
        self.norm_1 = SPADE(fin,fseg)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, seg):
        dx = self.conv_0(self.actvn(self.norm_0(x, seg)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg)))
        out = x + dx

        return out

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)

class SPADE(nn.Module):
    def __init__(self, norm_nc, label_nc):
        super().__init__()

        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)  #  * IN...

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128

        # pw = ks // 2
        self.mlp_shared = nn.Sequential(
            # nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.Conv2d(label_nc, nhidden, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)

    def forward(self, x, segmap):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out

class AdaConvDecoder(nn.Module):
    def __init__(self, style_channels, kernel_size):
        super().__init__()
        self.style_channels = style_channels
        self.kernel_size = kernel_size
        self.att0 = Self_Attn(512, nn.ReLU())
        self.att1 = Self_Attn(256, nn.ReLU())
        self.att2 = Self_Attn(128, nn.ReLU())
        self.att3 = Self_Attn(64, nn.ReLU())

        # Inverted VGG with first conv in each scale replaced with AdaConv
        n_convs = [2, 2, 2, 2]
        self.layers = nn.ModuleList([
            *self._make_layers(512, 256, n_convs=n_convs[0], kcov=True),
            *self._make_layers(256, 128, n_convs=n_convs[1], kcov=True),
            *self._make_layers(128, 64, n_convs=n_convs[2], kcov=True),
            *self._make_layers(64, 3, n_convs=n_convs[3], kcov=True, final_act=False, upsample=False)])

    def forward(self, content, w_style, parsing, style_attn):
        # Checking types is a bit hacky, but it works well.
        for module in self.layers:
            if isinstance(module, AdaConv2d):
                w_spatial = []
                w_pointwise = []
                bias = []
                temp = 0
                t = 0
                [b_size, f_size, h_size, w_size] = parsing.shape
                count = 0
                seg = F.interpolate(parsing, size=(h_size, w_size))
                for i in range(b_size):
                    for j in range(len(w_style)):
                        component_mask_area = torch.sum(seg.bool()[i, j])
                        if component_mask_area > 0:
                            w_spatial1, w_pointwise1, bias1 = w_style[j], w_style[j], w_style[j]
                            temp += module(content, w_spatial1, w_pointwise1, bias1, j)
                            count = count + 1
                content = content + temp
            else:
                content = module(content)
        return content

    def _make_layers(self, in_channels, out_channels, n_convs, final_act=True, upsample=True, kcov=False, ):
        layers = []
        for i in range(n_convs):
            last = i == n_convs - 1
            out_channels_ = out_channels if last else in_channels
            if i == 0 and kcov is True:
                layers += [
                    AdaConv2d(in_channels, out_channels_)]
            else:
                layers.append(ResBlocks(1, in_channels, 'none', 'relu'))
                layers.append(nn.Conv2d(in_channels, out_channels_, 3,
                                        padding=1))

            if not last or final_act:
                layers.append(nn.ReLU())

        #
        if upsample:
            layers.append(nn.Upsample(scale_factor=2, mode='nearest'))

        return layers

class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm='in', activation='relu', pad_type='zero', use_sn=False):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type, use_sn=use_sn)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)

class ResBlock(nn.Module):
    def __init__(self, dim, norm='in', activation='relu', pad_type='zero', use_sn=False):
        super(ResBlock, self).__init__()

        model = []
        model += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type, use_sn=use_sn)]
        model += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type, use_sn=use_sn)]

        model1 = []
        model1 += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type, use_sn=use_sn)]

        self.model = nn.Sequential(*model)
        self.model1 = nn.Sequential(*model1)

    def forward(self, x):
        residual = self.model1(x)
        out = self.model(x)
        out += residual
        return out

class Get_image(nn.Module):
    def __init__(self, input_dim, output_dim, activation='tanh'):
        super(Get_image, self).__init__()
        self.conv = Conv2dBlock(input_dim, output_dim, kernel_size=3, stride=1,
                                padding=1, pad_type='reflect', activation=activation)

    def forward(self, x):
        return self.conv(x)

class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero', dilation=1,
                 use_bias=True, use_sn=False):
        super(Conv2dBlock, self).__init__()
        self.use_bias = use_bias
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if use_sn:
            self.conv = spectral_norm(
                nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias, dilation=dilation))
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias, dilation=dilation)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'

class LayerNorm(nn.Module):
    def __init__(self, n_out, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.n_out = n_out
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(torch.ones(n_out, 1, 1))
            self.bias = nn.Parameter(torch.zeros(n_out, 1, 1))

    def forward(self, x):
        normalized_shape = x.size()[1:]
        if self.affine:
            return F.layer_norm(x, normalized_shape, self.weight.expand(normalized_shape),
                                self.bias.expand(normalized_shape))
        else:
            return F.layer_norm(x, normalized_shape)

class GlobalStyleEncoder(nn.Module):
    def __init__(self, in_shape, out_shape):
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        channels = in_shape[0]

        self.downscale = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.LeakyReLU(),
            nn.AvgPool2d(2, 2),
            #
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.LeakyReLU(),
            nn.AvgPool2d(2, 2),
            #
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.LeakyReLU(),
            nn.AvgPool2d(2, 2),
        )
        # self.avg = nn.AdaptiveAvgPool2d((4, 4))
        self.avg = nn.AdaptiveAvgPool1d(16)

        in_features = 512 * 16
        mid_features = 512 * 4
        out_features = 512
        self.fc0 = nn.Linear(in_features, mid_features)
        self.fc1 = nn.Linear(in_features, mid_features)
        self.fc2 = nn.Linear(in_features, mid_features)
        self.fc3 = nn.Linear(in_features, mid_features)
        self.fc4 = nn.Linear(in_features, mid_features)
        self.fc5 = nn.Linear(in_features, mid_features)

        self.fc00 = nn.Linear(mid_features, out_features)
        self.fc11 = nn.Linear(mid_features, out_features)
        self.fc22 = nn.Linear(mid_features, out_features)
        self.fc33 = nn.Linear(mid_features, out_features)
        self.fc44 = nn.Linear(mid_features, out_features)
        self.fc55 = nn.Linear(mid_features, out_features)

        self.fc000 = nn.Linear(out_features, out_features)
        self.fc111 = nn.Linear(out_features, out_features)
        self.fc222 = nn.Linear(out_features, out_features)
        self.fc333 = nn.Linear(out_features, out_features)
        self.fc444 = nn.Linear(out_features, out_features)
        self.fc555 = nn.Linear(out_features, out_features)

        self.p_fc1 = nn.Linear(512, 256)
        self.p_fc2 = nn.Linear(256, 128)
        self.p_fc3 = nn.Linear(128, 64)

    def forward(self, xs, parsing):
        [b_size, f_size, h_size, w_size] = xs[0].shape
        result = []
        codes_vector = torch.zeros((b_size, parsing.shape[1], 1, len(xs[0]), f_size, 1, 1), dtype=xs[0].dtype,
                                   device=xs[0].device)

        seg = F.interpolate(parsing, size=(h_size, w_size))
        for i in range(b_size):
            for j in range(parsing.shape[1]):
                component_mask_area = torch.sum(seg.bool()[i, j])
                if component_mask_area > 0:
                    ys = self.avg(
                        xs[(i + 1) * (j + 1) - 1].masked_select(seg.bool()[i, j]).reshape(f_size, component_mask_area))
                    ys = ys.reshape(len(xs[0]), -1)
                    w = self.__getattr__('fc' + str(j))(ys)
                    w = self.__getattr__('fc' + str(j) + str(j))(w)
                    w = self.__getattr__('fc' + str(j) + str(j) + str(j))(w)
                    w0 = w.reshape(len(xs[0]), self.out_shape[0], 1, 1)  # w (1,512,1,1)
                    w1 = self.p_fc1(w0.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # w (1,256,1,1)
                    w2 = self.p_fc2(w1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # w (1,128,1,1)
                    w3 = self.p_fc3(w2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # w (1,64,1,1)
                    s = [w0, w1, w2, w3]
                    result.append(s)
        return result

class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)  # 1*1 conv
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, style_attn):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(style_attn).view(m_batchsize, -1, width * height)
        # B X C X (*W*H)  torch.Size([1, 64, 1024])
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        # B X C x (*W*H)  torch.Size([1, 64, 1024])
        proj_query_hat = proj_query.sub(torch.mean(proj_query, dim=1))  # 通道进行mean
        # torch.Size([1, 64, 1024])
        proj_key_hat = proj_key.sub(torch.mean(proj_key, dim=1))
        pointwise_hat = torch.bmm(proj_query_hat.permute(0, 2, 1), proj_key_hat)

        proj_query_hat_n2 = torch.norm(proj_query_hat, p=2)  # 2范
        proj_key_hat_n2 = torch.norm(proj_key_hat, p=2)
        pointwise_n2 = proj_query_hat_n2 * proj_key_hat_n2
        energy = torch.div(pointwise_hat.float(), pointwise_n2)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)
        out = self.gamma * out + x
        return out, attention


#
# class ParsingDecoder(nn.Module):
#     def __init__(self, style_channels, kernel_size):
#         super().__init__()
#         self.style_channels1 = style_channels
#         self.kernel_size1 = kernel_size
#
#         # Inverted VGG with first conv in each scale replaced with AdaConv
#         n_convs1 = [1, 1, 1, 1]
#         self.layers1 = nn.ModuleList([
#             *self._make_layers1(512, 256, n_convs=n_convs1[0],),
#             *self._make_layers1(256, 128, n_convs=n_convs1[1],),
#             *self._make_layers1(128, 64, n_convs=n_convs1[2],),
#             *self._make_layers1(64, 20, n_convs=n_convs1[3],final_act=False, upsample=False)])
#
#     def forward(self, content_emb):
#         # Checking types is a bit hacky, but it works well.
#         for module in self.layers1:
#             # print('content_emb',content_emb.shape) torch.Size([1, 512, 32, 32])
#             output = module(content_emb)
#             print('content_emb', content_emb.shape)
#             print('output',output.shape)
#         return output
#
#     def _make_layers1(self, in_channels, out_channels, n_convs, final_act=False, upsample=False,):
#         layers = []
#         # for i in range(n_convs):
#             # last = i == n_convs - 1
#             # out_channels_ = out_channels if last else in_channels
#         out_channels_ = out_channels
#         layers.append(ResBlocks(1, in_channels, 'none', 'relu'))
#         layers.append(nn.Conv2d(in_channels, out_channels_, 3,
#                                         padding=1))
#         if final_act:
#             layers.append(nn.ReLU())
#         if upsample:
#             layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
#         return layers

# 备选
class U_Net(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(U_Net, self).__init__()

        # self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=inchannel, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, outchannel, kernel_size=1, stride=1, padding=0)
        self.tanh = nn.Tanh()

    def forward(self, x):  # x:512*256*256
        # encoding path
        x1 = self.Conv1(x)  # 64*256*256
        x2 = self.Maxpool(x1)  # 64*128*128
        x2 = self.Conv2(x2)  # 128*128*128

        x3 = self.Maxpool(x2)  # 128*64*64
        x3 = self.Conv3(x3)  # 256*64*64

        x4 = self.Maxpool(x3)  # 256*32*32
        x4 = self.Conv4(x4)  # 512*32*32

        x5 = self.Maxpool(x4)  # 512*16*16
        x5 = self.Conv5(x5)  # 1024*16*16

        # decoding + concat path
        d5 = self.Up5(x5)  # 512*32*32
        d5 = torch.cat((x4, d5), dim=1)  # 1024*32*32

        d5 = self.Up_conv5(d5)  # 512*32*32

        d4 = self.Up4(d5)  # 256*64*64
        d4 = torch.cat((x3, d4), dim=1)  # 512*64*64
        d4 = self.Up_conv4(d4)  # 256*64*64

        d3 = self.Up3(d4)  # 128*128*128
        d3 = torch.cat((x2, d3), dim=1)  # 256*128*128
        d3 = self.Up_conv3(d3)  # 128*128*128

        d2 = self.Up2(d3)  # 64*256*256
        d2 = torch.cat((x1, d2), dim=1)  # 128*256*256
        d2 = self.Up_conv2(d2)  # 64*256*256

        d1 = self.Conv_1x1(d2)  # 3*256*256
        d1 = self.tanh(d1)
        # print('Unet forward')
        return d1


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.BatchNorm2d(ch_out),
            nn.InstanceNorm2d(ch_out, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.BatchNorm2d(ch_out),
            nn.InstanceNorm2d(ch_out, affine=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.BatchNorm2d(ch_out),
            nn.InstanceNorm2d(ch_out, affine=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x