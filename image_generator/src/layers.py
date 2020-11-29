from torchvision import models
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


class SPADE(nn.Module):
    def __init__(self, x_dim, y_mod_dim=128, norm_type='instance', ks=3):
        """Modified from https://github.com/NVlabs/SPADE/blob/master/models/networks/normalization.py"""
        super().__init__()

        if norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(x_dim, affine=False)
        elif norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(x_dim, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % norm_type)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128

        pw = ks // 2
        self.shared = nn.Sequential(
            nn.Conv2d(y_mod_dim, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.gamma = nn.Conv2d(nhidden, x_dim, kernel_size=ks, padding=pw)
        self.beta = nn.Conv2d(nhidden, x_dim, kernel_size=ks, padding=pw)

    def forward(self, x, y):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        # y = F.interpolate(y, size=x.size()[2:], mode='nearest')
        y = F.interpolate(y, size=x.size()[2:], mode='bilinear', align_corners=False)
        actv = self.shared(y)
        gamma = self.gamma(actv)
        beta = self.beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta
        return out


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=False):
        if noise:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()
            return image + self.weight * noise
        else:
            return image


class GeneratorResidualBlock(nn.Module):
    def __init__(self, n_in, n_out, y_mod_dim=128, upscale=True, norm_type='spade_in',
                 SN=nn.utils.spectral_norm):
        super().__init__()
        if upscale:
            upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        else:
            upsample = nn.Identity()

        norm = {
            'spade_in': partial(SPADE, norm_type='instance'),
        }[norm_type]

        self.cbn1 = norm(n_in, y_mod_dim=y_mod_dim)
        self.relu1 = nn.LeakyReLU(0.2)
        self.upsample = upsample
        self.conv1 = SN(nn.Conv2d(n_in, n_out, 3, padding=1))
        self.noise1 = NoiseInjection()
        self.cbn2 = norm(n_out, y_mod_dim=y_mod_dim)
        self.relu2 = nn.LeakyReLU(0.2)
        self.conv2 = SN(nn.Conv2d(n_out, n_out, 3, padding=1))
        self.noise2 = NoiseInjection()

        self.res_branch = nn.Sequential(
            upsample,
            SN(nn.Conv2d(n_in, n_out, 1, padding=0))
        )

    def forward(self, x, y=None, noise=False):
        # y: z + coordinate

        h = self.cbn1(x, y)
        h = self.noise1(h, noise)

        h = self.relu1(h)
        h = self.upsample(h)
        h = self.conv1(h)

        h = self.cbn2(h, y)
        h = self.noise2(h, noise)

        h = self.relu2(h)
        h = self.conv2(h)

        res = self.res_branch(x)

        out = h + res

        return out


class ToRGB(nn.Module):
    def __init__(self, in_channel, target_size=224):
        super().__init__()

        self.conv = nn.Conv2d(in_channel, 3, 3, padding=1)
        # self.tanh = nn.Tanh()
        self.up = nn.Upsample(
            size=(target_size, target_size),
            mode='bilinear', align_corners=False)

    def forward(self, x, up=True):
        h = self.conv(x)
        # h = self.tanh(h)
        if up:
            out = self.up(h)
            return out
        return h


class Generator(nn.Module):
    def __init__(self, emb_dim=2048, mod_dim=128, base_dim=64, n_channel=3, target_size=16, extra_layers=0,
                 init_H=8, init_W=8, norm_type='spade_in', SN=True, codebook_dim=256):
        super().__init__()

        self.init_H = init_H
        self.init_W = init_W
        self.target_size = target_size
        self.norm_type = norm_type
        self.SN = SN
        self.emb_dim = emb_dim

        self.bottleneck_emb = nn.Sequential(
            nn.Conv2d(emb_dim, codebook_dim, 1, padding=0),
            nn.Tanh()
        )

        if SN:
            SN = nn.utils.spectral_norm
        else:
            def SN(x): return x

        upscale_ratio = target_size // init_H

        # n_init = base_dim * upscale_ratio

        resolution_channels = {
            7: min(512, base_dim),
            14: min(512, base_dim),
            28: min(512, base_dim),
            56: min(512, base_dim),
            112: min(256, base_dim),
            224: min(128, base_dim),

            8: min(512, base_dim),
            16: min(512, base_dim),
            32: min(512, base_dim),
            64: min(512, base_dim),
            128: min(256, base_dim),
            256: min(128, base_dim),
        }
        n_init = base_dim

        self.learned_init_conv = nn.Sequential(
            SN(nn.Conv2d(codebook_dim, n_init, 3, padding=1, groups=4)),
        )

        mod_dim = n_init
        self.style_init_conv = nn.Sequential(
            SN(nn.Conv2d(codebook_dim, mod_dim, 3, padding=1, groups=4)),
        )

        n_upscale_resblocks = int(np.log2(upscale_ratio))
        resblocks = []

        to_RGB_blocks = []
        res = init_H
        # upscale resblocks
        for i in range(n_upscale_resblocks):
            n_in = resolution_channels[res]
            res = res * 2
            n_out = resolution_channels[res]

            resblocks.append(GeneratorResidualBlock(n_in, n_out, mod_dim,
                                                    norm_type=norm_type, SN=SN))

            to_RGB_blocks.append(ToRGB(n_out, self.target_size))

        # extra resblocks (no upscales)
        for _ in range(extra_layers):
            n_in = resolution_channels[res]
            # res = res * 2
            n_out = resolution_channels[res]
            resblocks.append(GeneratorResidualBlock(n_in, n_out, mod_dim,
                                                    upscale=False,
                                                    norm_type=norm_type, SN=SN))

            to_RGB_blocks.append(ToRGB(n_out, self.target_size))

        self.resblocks = nn.ModuleList(resblocks)
        self.to_RGB_blocks = nn.ModuleList(to_RGB_blocks)

        self.last = nn.Sequential(
            nn.Tanh(),
        )

        self.init_parameter()

    def forward(self, emb, train=True):
        """
        emb: [B, init_H, init_W, 2048]

        out: [B, 3, target_size, target_size]
        """
        B = emb.size(0)

        if emb.size()[1:] == (self.init_H, self.init_W, self.emb_dim):
            # [B, 2048, init_H, init_W]
            emb = emb.permute(0, 3, 1, 2)

        # [B, n_init, init_H, init_W]
        emb = self.bottleneck_emb(emb)

        h = self.learned_init_conv(emb)
        y = self.style_init_conv(emb)

        out = torch.zeros(B, 3, self.target_size, self.target_size).to(h.device)

        # [B, base_dim, target_size, target_size]
        for i, resblock in enumerate(self.resblocks):

            h = resblock(h, y, noise=train)
            upsample = i+1 < len(self.resblocks)
            RGB_out = self.to_RGB_blocks[i](h, up=upsample)
            out = out + RGB_out

        out = self.last(out)

        return out

    def init_parameter(self):
        for k in self.named_parameters():
            if k[1].dim() > 1:
                torch.nn.init.orthogonal_(k[1])
            if k[0][-4:] == 'bias':
                torch.nn.init.constant_(k[1], 0)

class ResBlock(nn.Module):
    def __init__(self, in_channel, channel=32, SN=False):
        super().__init__()

        if SN:
            SN = nn.utils.spectral_norm
        else:
            def SN(x): return x

        self.conv = nn.Sequential(
            nn.ReLU(inplace=True),
            SN(nn.Conv2d(in_channel, channel, 3, padding=1)),
            nn.ReLU(inplace=True),
            SN(nn.Conv2d(channel, in_channel, 1)),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out


class ResNetEncoder(nn.Module):
    def __init__(self, model='resnet101'):
        super().__init__()
        if model == 'resnet101':
            self.net = models.resnet101(pretrained=True, progress=True)
        elif model == 'resnet50':
            self.net = models.resnet50(pretrained=True, progress=True)

    def forward(self, x, obj_class=False, pool=True, last_feat_only=False, upscale_ratio=16):
        """
        Extract feature with resnet
        https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py#L197
        """
        x = self.net.conv1(x) # [H/2, W/2]
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x) # [H/4, W/4]

        layer1 = self.net.layer1(x)       # [H/4, W/4]
        layer2 = self.net.layer2(layer1)  # [H/8, W/8]
        layer3 = self.net.layer3(layer2)  # [H/16, W/16]
        layer4 = self.net.layer4(layer3)  # [H/32, W/32]

        layers = layer1, layer2, layer3, layer4

        if last_feat_only:
            # [7 -> 224]
            # [8 -> 256]
            if upscale_ratio == 32:
                # [B, 2048, 14, 14]
                downsample_layer = layers[3]

            # [14 -> 224]
            # [7 -> 112]
            # [8 -> 128]
            elif upscale_ratio == 16:
                # [B, 1024, 14, 14]
                downsample_layer = layers[2]

            # [28 -> 224]
            # [14 -> 112]
            # [8 -> 64]
            elif upscale_ratio == 8:
                downsample_layer = layers[1]

            # [56 -> 224]
            # [28 -> 112]
            # [8 -> 32]
            elif upscale_ratio == 4:
                downsample_layer = layers[0]

            return downsample_layer

        logit = None
        if pool:
            avg_pool_feature = self.net.avgpool(layer4)  # [B, 2048, 1, 1]
            last_feat = torch.flatten(avg_pool_feature, 1)  # [B, 2048]
            if obj_class:
                logit = self.net.fc(last_feat)  # [B, 1000]
        else:
            last_feat = layer4.permute(0, 2, 3, 1)  # [B, H/32 W/32, 2048]
            if obj_class:
                logit = self.net.fc(last_feat)  # [B, H/32, W/32, 1000]

        return layers, last_feat, logit


class DiscriminatorResidualBlock(nn.Module):
    def __init__(self, n_in, n_out, downsample=True, first_relu=True, activation='leakyrelu',
                 SN=nn.utils.spectral_norm):
        super().__init__()
        if downsample:
            downsample = nn.AvgPool2d(2, 2)
        else:
            downsample = nn.Identity()

        if first_relu:
            if activation == 'leakyrelu':
                self.relu1 = nn.LeakyReLU(0.2, False)
            else:
                self.relu1 = nn.ReLU()
        else:
            self.relu1 = nn.Identity()
        self.conv1 = SN(nn.Conv2d(n_in, n_out, 3, padding=1))
        self.norm1 = nn.InstanceNorm2d(n_out, affine=False)
        if activation == 'leakyrelu':
            self.relu2 = nn.LeakyReLU(0.2, False)
        else:
            self.relu2 = nn.ReLU()
        self.conv2 = SN(nn.Conv2d(n_out, n_out, 3, padding=1))
        self.downsample = downsample

        self.res_branch = nn.Sequential(
            downsample,
            SN(nn.Conv2d(n_in, n_out, 1, padding=0))
        )

    def forward(self, x):
        x = self.relu1(x)
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.relu2(h)
        h = self.conv2(h)
        h = self.downsample(h)

        res = self.res_branch(x)

        out = h + res
        return out


class Discriminator(nn.Module):
    def __init__(self, base_dim=512, emb_dim=2048, n_channel=3,
                 target_size=160, extra_layers=0,
                 init_H=8, init_W=8, SN=True, ACGAN=False, n_classes=10000):
        super().__init__()

        y_dim = emb_dim

        self.SN = SN
        self.init_H = init_H
        self.init_W = init_W

        if SN:
            SN = nn.utils.spectral_norm
        else:
            def SN(x): return x

        downsample_ratio = target_size // init_H
        n_downsample_resblocks = int(np.log2(downsample_ratio))
        # proj_dim = base_dim * 2**(n_downsample_resblocks + 1 + extra_layers)
        resblocks = []

        resolution_channels = {
            7: min(512, base_dim),
            14: min(512, base_dim),
            28: min(512, base_dim),
            56: min(512, base_dim),
            112: min(256, base_dim),
            224: min(128, base_dim),

            8: min(512, base_dim),
            16: min(512, base_dim),
            32: min(512, base_dim),
            64: min(512, base_dim),
            128: min(256, base_dim),
            256: min(128, base_dim),
        }
        res = target_size

        # extra resblocks (no downsample)
        # n_dim = base_dim
        for i in range(extra_layers):
            downsample = False
            first_relu = False if i == 0 else True

            n_in = resolution_channels[res]
            # res = res // 2
            n_out = resolution_channels[res]
            if i == 0:
                n_in = 3
            if i == extra_layers - 1:
                n_out = resolution_channels[res]
            resblocks.append(
                DiscriminatorResidualBlock(n_in, n_out, downsample, first_relu, SN=SN))

        # downsample resblocks
        for i in range(n_downsample_resblocks):
            downsample = True
            first_relu = False

            n_in = resolution_channels[res]
            res = res // 2
            n_out = resolution_channels[res]

            if len(resblocks) == 0 and i == 0:
                n_in = 3

            if extra_layers > 0 or i > 0:
                first_relu = True

            resblocks.append(
                DiscriminatorResidualBlock(n_in, n_out, downsample, first_relu, SN=SN))

        n_dim = n_out

        # last resblock (no downsample)
        resblocks.append(
            DiscriminatorResidualBlock(n_dim, n_dim, False, True, SN=SN))

        # assert n_dim == proj_dim

        self.resblocks = nn.ModuleList(resblocks)

        self.relu = nn.ReLU()
        # self.global_pool = nn.AdaptiveMaxPool2d(1)

        # self.adv_out = SpectralNorm(nn.Linear(n_dim, 1))
        self.adv_out = SN(nn.Conv2d(n_dim, 1, 3, padding=1))

        self.ACGAN = ACGAN
        if ACGAN:
            self.emb_proj = nn.Conv2d(n_dim, emb_dim, 1, padding=0)
            self.emb_classifier = nn.Linear(emb_dim, n_classes)
            self.n_classes = n_classes

        # projection discriminator
        else:
            self.y_proj = SN(nn.Conv2d(
                y_dim, n_dim // 2, 1, padding=0, bias=False))
            self.h_proj = SN(nn.Conv2d(
                n_dim, n_dim // 2, 1, padding=0, bias=False))

        self.init_parameter()

    def forward(self, x, y, output_layers=False):
        """
        x: [B, 3, H, W]
        y: [B, init_H, init_W, 2048]
        """
        D_layers = []
        h = x
        for resblock in self.resblocks:
            h = resblock(h)
            D_layers.append(h)

        # [B, n_dim, init_H, init_W]
        h = self.relu(h)

        B, n_dim, init_H, init_W = h.size()
        # [B, 1, init_H x init_W]
        adv_out = self.adv_out(h)
        # [B, 1]
        adv_out = adv_out.mean(dim=(2, 3))

        if y.size()[1:3] == (self.init_H, self.init_W):
            # [B, 2048, init_H, init_W]
            y = y.permute(0, 3, 1, 2).contiguous()

        if self.ACGAN:
            emb = self.emb_proj(h)
            emb = emb.permute(0,2,3,1)
            cls_logit = self.emb_classifier(emb)
            cls_logit = cls_logit.view(B * init_H * init_W, self.n_classes)

            if output_layers:
                return adv_out, D_layers, cls_logit
            else:
                return adv_out, cls_logit

        # projection discriminator
        else:
            # [B, n_dim //2 , init_H, init_W]
            y_proj = self.y_proj(y)
            # [B, n_dim //2 , init_H, init_W]
            h_proj = self.h_proj(h)
            # [B, 1, init_H, init_W]
            proj_out = torch.mul(h_proj, y_proj).sum(dim=1, keepdim=True)
            # [B, 1]
            proj_out = proj_out.mean(dim=(2, 3))

            out = adv_out + proj_out

            if output_layers:
                return out, D_layers
            else:
                return out

    def init_parameter(self):
        for k in self.named_parameters():
            if k[1].dim() > 1:
                torch.nn.init.orthogonal_(k[1])
            if k[0][-4:] == 'bias':
                torch.nn.init.constant_(k[1], 0)
