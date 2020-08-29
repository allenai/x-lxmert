from copy import deepcopy
from collections import deque
import os
import random
from tqdm import tqdm
from pathlib import Path
from itertools import chain
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import save_image

import apex
from apex import amp
from torch.utils.tensorboard import SummaryWriter

from data_utils import grid_view, original_view, box_position


class LossMeter(object):
    def __init__(self, maxlen=100):
        """Computes and stores the running average"""
        self.vals = deque([], maxlen=maxlen)

    def update(self, new_val):
        self.vals.append(new_val)

    @property
    def val(self):
        return sum(self.vals) / len(self.vals)

    def __repr__(self):
        return str(self.val)


class Trainer():

    def __init__(self, config, E, G, D, Emb,
                 g_optim, d_optim,
                 train_loader, val_loader, logger=None):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = logger

        self.E = E
        self.G = G
        self.D = D
        self.Emb = Emb

        self.g_optim = g_optim
        self.d_optim = d_optim

        if self.config.gpu == 0:
            self.log_dir = config.log_dir
            self.img_log_dir = self.log_dir.joinpath('images')
            self.img_log_dir.mkdir(exist_ok=True, parents=True)
            self.ckpt_dir = self.log_dir.joinpath('ckpt')
            self.ckpt_dir.mkdir(exist_ok=True, parents=True)
            assert self.logger is not None
            self.writer = SummaryWriter(str(self.log_dir))

            hparam_dict = {}
            for k, v in self.config.__dict__.items():
                if type(v) in [int, float, str, bool, torch.Tensor]:
                    hparam_dict[k] = v
            metric_dict = {}

            self.writer.add_hparams(hparam_dict, metric_dict)
        else:
            assert self.logger is None

        self.n_grid = self.config.n_grid

        # For debugging
        self.debug_real_img = None
        self.debug_z = None
        self.debug_img_grids_feat = None

    def norm(self, x):
        """(0, 1) => (-1, 1)"""
        out = 2 * x - 1
        return out.clamp(-1, 1)

    def denorm(self, x):
        """(-1, 1) => (0, 1)"""
        out = (x + 1) / 2
        return out.clamp(0, 1)

    def imagenet_norm(self, x):
        # return (x - self.imagenet_mean) / self.imagenet_std
        imagenet_mean = torch.tensor(
            [0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        imagenet_std = torch.tensor(
            [0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        return (x - imagenet_mean) / imagenet_std

    def imagenet_denorm(self, x):
        # return x * self.imagenet_std + self.imagenet_mean
        imagenet_mean = torch.tensor(
            [0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        imagenet_std = torch.tensor(
            [0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        return x * imagenet_std + imagenet_mean

    def forward(self, data, train=True, verbose=False):
        out_dict = {}

        real_img = data['img']
        if self.config.clustering:
            cluster_ids = data['cluster_id']

        real_img = real_img.cuda(non_blocking=True).float() / 255
        B, C, H, W = real_img.size()

        device = real_img.device
        dtype = torch.float

        recon_loss = torch.zeros(1, dtype=dtype).to(device).squeeze()
        feat_loss = torch.zeros_like(recon_loss)
        obj_loss = torch.zeros_like(recon_loss)
        g_loss_fake = torch.zeros_like(recon_loss)
        g_loss_feat_match = torch.zeros_like(recon_loss)
        g_loss_cluster = torch.zeros_like(recon_loss)

        upscale_ratio = self.config.resize_target_size // self.config.n_grid
        resized_target_img = F.interpolate(
            real_img,
            size=(self.config.resize_target_size, self.config.resize_target_size),
            mode='bilinear', align_corners=False)

        #############################
        # G Update
        #############################
        if train:
            self.G.train()
            if self.config.gan:
                self.D.eval()

        if verbose:
            print('Before E')


        else:
            if self.config.clustering:
                cluster_ids = cluster_ids.cuda(non_blocking=True).view(
                    B * self.n_grid * self.n_grid)
                # [B x n_grid x n_grid, feat_dim]
                img_grids_feat = self.Emb(cluster_ids)

                img_grids_feat = img_grids_feat.view(B, self.n_grid, self.n_grid, -1)
                img_grids_feat = img_grids_feat.permute(0,3,1,2)

            else:
                downsample_ratio = self.config.resize_input_size // self.config.n_grid

                # [B, feat_dim, n_grid, n_grid]
                img_grids_feat = self.E(self.imagenet_norm(real_img),
                                        last_feat_only=True,
                                        upscale_ratio=downsample_ratio)

        # [B, feat_dim, n_grid, n_grid]
        assert img_grids_feat.size(0) == B
        assert img_grids_feat.size()[2:] == (self.n_grid, self.n_grid)

        if verbose:
            print('After E')

        img_grids_feat = img_grids_feat.detach()

        # [B, 3, resize_target_size, resize_target_size]
        recon_img = self.G(img_grids_feat)

        if verbose:
            print('After G')

        if self.config.recon_loss_lambda:
            recon_loss = F.smooth_l1_loss(
                recon_img, self.norm(resized_target_img))

        if self.config.obj_loss_lambda or self.config.feat_loss_lambda:
            self.E.eval()
            recon_out = self.E(self.imagenet_norm(self.denorm(recon_img)),
                            pool=self.config.feat_pool,
                            obj_class=self.config.obj_loss_lambda,
                            last_feat_only=not self.config.all_layers,
                            upscale_ratio=upscale_ratio
                            )
            if self.config.all_layers:
                recon_layers, recon_last_feat, recon_logit = recon_out
            else:
                recon_last_feat = recon_out
            with torch.no_grad():
                tgt_out = self.E(self.imagenet_norm(resized_target_img),
                                pool=self.config.feat_pool,
                                obj_class=self.config.obj_loss_lambda,
                                last_feat_only=not self.config.all_layers,
                                upscale_ratio=upscale_ratio
                                )

                if self.config.all_layers:
                    tgt_layers, tgt_last_feat, tgt_logit = tgt_out
                else:
                    tgt_last_feat = tgt_out

            if self.config.feat_loss_lambda:
                if self.config.all_layers:
                    for recon_layer, tgt_layer in zip(recon_layers, tgt_layers):
                        _feat_loss = F.smooth_l1_loss(recon_layer, tgt_layer.detach(), reduction='none')
                        _feat_loss = _feat_loss.mean(1).mean(dim=(1,2)).mean()
                        feat_loss = feat_loss + _feat_loss
                else:
                    feat_loss = F.smooth_l1_loss(recon_last_feat, tgt_last_feat.detach(), reduction='none')
                    feat_loss = feat_loss.mean(-1).mean(dim=(1, 2)).mean()
            if self.config.obj_loss_lambda:
                if self.config.feat_pool:
                    obj_loss = F.kl_div(F.log_softmax(recon_logit, dim=1),
                                        F.softmax(tgt_logit, dim=1).detach(),
                                        reduction='none')
                    assert obj_loss.size(1) == 1000, obj_loss.size()
                    obj_loss = obj_loss.mean(1).mean(0)
                    # obj_loss = obj_loss.sum(1).mean(0)
                    assert obj_loss.item() >= 0, obj_loss
                else:
                    obj_loss = F.kl_div(F.log_softmax(recon_logit, dim=3),
                                        F.softmax(tgt_logit, dim=3).detach(),
                                        reduction='none')
                    _H = obj_loss.size(1)
                    _W = obj_loss.size(2)
                    assert obj_loss.size(3) == 1000, obj_loss.size()
                    # obj_loss = obj_loss.sum(3).view(
                    #     B, _H * _W).mean(1).mean(0)
                    obj_loss = obj_loss.mean(3).view(
                        B, _H * _W).mean(1).mean(0)
                    assert obj_loss.item() >= 0, obj_loss

        if self.config.gan:
            fake_img = recon_img
            if self.config.ACGAN:
                D_fake, D_layers_fake, D_cls_fake = self.D(fake_img, img_grids_feat, True)
            else:
                D_fake, D_layers_fake = self.D(fake_img, img_grids_feat, True)

            if self.config.hinge:
                g_loss_fake = -D_fake.mean()
            else:
                g_loss_fake = -F.logsigmoid(D_fake).mean()

            if self.config.gan_feat_match_lambda:
                if self.config.ACGAN:
                    D_real, D_layers_real, _D_cls_real = self.D(
                        self.norm(resized_target_img), img_grids_feat, True)
                else:
                    D_real, D_layers_real = self.D(
                        self.norm(resized_target_img), img_grids_feat, True)

                for recon_layer, tgt_layer in zip(
                        D_layers_fake[:self.config.gan_feat_match_layers],
                        D_layers_real[:self.config.gan_feat_match_layers]):
                    g_feat_loss = F.smooth_l1_loss(recon_layer, tgt_layer, reduction='none')
                    g_feat_loss = g_feat_loss.mean(1).mean(dim=(1,2)).mean()
                    g_loss_feat_match = g_loss_feat_match + g_feat_loss

            if self.config.ACGAN:
                g_loss_cluster = F.cross_entropy(D_cls_fake, cluster_ids)

        g_loss = recon_loss + self.config.recon_loss_lambda + \
            feat_loss * self.config.feat_loss_lambda + \
            obj_loss * self.config.obj_loss_lambda + \
            g_loss_fake * self.config.gan_loss_lambda + \
            g_loss_feat_match * self.config.gan_feat_match_lambda + \
            g_loss_cluster * self.config.gan_loss_cluster_lambda

        if verbose:
            print('After Loss')


        isnan = False
        if torch.isnan(recon_loss):
            print('recon loss is nan')
            isnan = True
        if torch.isnan(feat_loss):
            print('feat loss is nan')
            isnan = True
        if torch.isnan(obj_loss):
            print('obj loss is nan')
            isnan = True
        if torch.isnan(g_loss_fake):
            print('g loss is nan')
            isnan = True
        if torch.isnan(g_loss_feat_match):
            print('g_loss_feat_match is nan')
            isnan = True

        if isnan:
            exit()

        if train:
            if verbose:
                print('Before backprop')

            self.g_optim.zero_grad()
            if self.config.mixed_precision:
                with amp.scale_loss(g_loss, self.g_optim) as g_scaled_loss:
                    g_scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    amp.master_params(self.g_optim), self.config.grad_clip_norm)
            else:
                g_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.G.parameters(), self.config.grad_clip_norm)
            self.g_optim.step()

            if verbose:
                print('After backprop')

        out_dict['recon_loss'] = recon_loss.item()
        out_dict['feat_loss'] = feat_loss.item()
        out_dict['obj_loss'] = obj_loss.item()
        out_dict['g_loss_fake'] = g_loss_fake.item()
        out_dict['g_loss_feat_match'] = g_loss_feat_match.item()
        out_dict['g_loss_cluster'] = g_loss_cluster.item()
        out_dict['g_loss'] = g_loss.item()

        #############################
        # D Update
        #############################

        if self.config.gan:
            if train:
                self.G.eval()
                self.D.train()

            resized_target_img = resized_target_img.detach().clone()
            img_grids_feat = img_grids_feat.detach().clone()

            if self.config.ACGAN:
                D_fake, D_cls_fake = self.D(fake_img.contiguous(), img_grids_feat, False)
                D_real, D_cls_real = self.D(self.norm(resized_target_img),img_grids_feat, False)

                d_loss_cluster_fake = F.cross_entropy(D_cls_fake, cluster_ids)
                d_loss_cluster_real = F.cross_entropy(D_cls_real, cluster_ids)

            else:
                D_fake = self.D(fake_img.contiguous(), img_grids_feat, False)
                D_real = self.D(self.norm(resized_target_img),img_grids_feat, False)

            if self.config.hinge:
                d_loss_real = F.relu(1.0 - D_real).mean()
                d_loss_fake = F.relu(1.0 + D_fake).mean()
            else:
                d_loss_real = -F.logsigmoid(D_real).mean()
                # d_loss_fake = F.logsigmoid(D_fake).mean()
                d_loss_fake = -(1 - torch.sigmoid(D_fake)).log().mean()

            d_loss = d_loss_real + d_loss_fake

            if self.config.ACGAN:
                d_loss = d_loss + self.config.gan_loss_cluster_lambda * (d_loss_cluster_fake + d_loss_cluster_real)

            if torch.isnan(d_loss_real):
                print('d_loss_real is nan')
                isnan = True
            if torch.isnan(d_loss_fake):
                print('d_loss_fake is nan')
                isnan = True
            if isnan:
                exit()

            if train:
                self.d_optim.zero_grad()
                if self.config.mixed_precision:
                    with amp.scale_loss(d_loss, self.d_optim) as d_scaled_loss:
                        d_scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        amp.master_params(self.d_optim), self.config.grad_clip_norm)
                else:
                    d_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.D.parameters(), self.config.grad_clip_norm)
                self.d_optim.step()

            out_dict['d_loss_fake'] = d_loss_fake.item()
            out_dict['d_loss_real'] = d_loss_real.item()
            out_dict['d_loss'] = d_loss.item()

            out_dict['d_fake'] = D_fake.mean().item()
            out_dict['d_real'] = D_real.mean().item()

            if self.config.ACGAN:
                out_dict['d_loss_cluster_fake'] = d_loss_cluster_fake.mean().item()
                out_dict['d_loss_cluster_real'] = d_loss_cluster_real.mean().item()

        return out_dict

    def train(self):

        d_lr = self.config.d_lr
        g_lr = self.config.g_lr
        n_steps_per_epoch = len(self.train_loader)

        if self.config.gpu == 0:
            print(f'# steps per epoch {n_steps_per_epoch}')
            self.logger.info(f'# steps per epoch {n_steps_per_epoch}')

        for epoch in range(self.config.epochs):

            if self.config.classifier:
                self.E.eval()

            if self.config.distributed:
                self.train_loader.sampler.set_epoch(epoch)

            if self.config.gpu == 0:
                self.pbar = tqdm(
                    ncols=self.config.tqdm_len,
                    # dynamic_ncols=True,
                    total=n_steps_per_epoch)

                recon_loss_meter = LossMeter()
                feat_loss_meter = LossMeter()
                obj_loss_meter = LossMeter()

                g_loss_fake_meter = LossMeter()
                g_loss_feat_match_meter = LossMeter()

                g_loss_cluster_meter = LossMeter()

                d_loss_fake_meter = LossMeter()
                d_loss_real_meter = LossMeter()

                d_loss_cluster_fake_meter = LossMeter()
                d_loss_cluster_real_meter = LossMeter()

                d_fake_meter = LossMeter()
                d_real_meter = LossMeter()

                g_loss_meter = LossMeter()
                d_loss_meter = LossMeter()

            for step_i, data in enumerate(self.train_loader):

                B = len(data['img'])
                # if self.config.gpu == 0:
                #     print(f'before forward prop - step{step_i}')
                # with torch.autograd.set_detect_anomaly(True):
                #     out_dict = self.forward(data, train=True, verbose=self.config.gpu==0)
                out_dict = self.forward(data, train=True)

                #############################
                # Logging
                #############################
                if self.config.gpu == 0:
                    # print(f'after forward prop - step{step_i}')

                    if self.debug_real_img is None:
                        data = next(iter(self.val_loader))

                        real_img = data['img']
                        if self.config.clustering:
                            cluster_ids = data['cluster_id']
                            self.debug_cluster_ids = deepcopy(cluster_ids)
                        real_img = real_img.cuda(
                            non_blocking=True).float() / 255

                        self.debug_real_img = deepcopy(real_img)

                    if self.config.recon_loss_lambda:
                        recon_loss_meter.update(out_dict['recon_loss'])

                    if self.config.feat_loss_lambda:
                        feat_loss_meter.update(out_dict['feat_loss'])
                    if self.config.obj_loss_lambda:
                        obj_loss_meter.update(out_dict['obj_loss'])


                    if self.config.gan:

                        if self.config.gan_feat_match_lambda:
                            g_loss_feat_match_meter.update(
                                out_dict['g_loss_feat_match'])

                        g_loss_fake_meter.update(out_dict['g_loss_fake'])

                        d_loss_fake_meter.update(out_dict['d_loss_fake'])
                        d_loss_real_meter.update(out_dict['d_loss_real'])

                        d_real_meter.update(out_dict['d_real'])
                        d_fake_meter.update(out_dict['d_fake'])

                        d_loss_meter.update(out_dict['d_loss'])
                        g_loss_meter.update(out_dict['g_loss'])

                        if self.config.ACGAN:
                            g_loss_cluster_meter.update(out_dict['g_loss_cluster'])
                            d_loss_cluster_real_meter.update(out_dict['d_loss_cluster_real'])
                            d_loss_cluster_fake_meter.update(out_dict['d_loss_cluster_fake'])

                    if step_i > 0:
                        log_str = f'Epoch {epoch} | Step {step_i} | Batch {B} | LR G: {g_lr:.1e}'
                        if self.config.gan:
                            log_str += f' D: {d_lr:.1e}'
                            log_str += f' | (G) fake: {g_loss_fake_meter.val:.3f}'
                            if self.config.gan_feat_match_lambda:
                                log_str += f' feat_match: {g_loss_feat_match_meter.val:.3f}'
                            if self.config.ACGAN:
                                log_str += f' clst_fake: {d_loss_cluster_fake_meter.val:.3f}'
                                log_str += f' clst_real: {d_loss_cluster_real_meter.val:.3f}'
                            log_str += f' (D) fake: {d_loss_fake_meter.val:.3f} real: {d_loss_real_meter.val:.3f}'
                            log_str += f' | D(fake) {d_fake_meter.val:.3f} D(real) {d_real_meter.val:.3f}'
                            log_str += f' | G loss: {g_loss_meter.val:.3f} D loss {d_loss_meter.val:.3f}'
                        log_str += f' |'
                        if self.config.recon_loss_lambda:
                            log_str += f' recon loss: {recon_loss_meter.val:.3f}'
                        if self.config.feat_loss_lambda:
                            log_str += f' feat loss: {feat_loss_meter.val:.3f}'
                        if self.config.obj_loss_lambda:
                            log_str += f' obj loss: {obj_loss_meter.val:.3f}'

                        self.pbar.set_description(log_str)

                        if (step_i % self.config.log_step == 0) or (step_i + 1 == n_steps_per_epoch):
                            self.pbar.write(log_str)
                            self.logger.info(log_str)

                        if (step_i % self.config.log_img_step == 0) or (step_i + 1 == n_steps_per_epoch):

                            self.G.eval()
                            if self.config.classifier:
                                self.E.eval()
                            self.sample(epoch, step_i)

                            if self.config.sample_from_feat:
                                self.sample(epoch, step_i, from_feat=True)

                            # print('image sampling done')

                    self.pbar.update(1)

            if self.config.gpu == 0:
                self.pbar.close()

                if self.config.recon_loss_lambda:
                    self.writer.add_scalar(
                        'Train/recon_loss',
                        recon_loss_meter.val, epoch)
                if self.config.feat_loss_lambda:
                    self.writer.add_scalar(
                        'Train/feat_loss',
                        feat_loss_meter.val, epoch)
                if self.config.obj_loss_lambda:
                    self.writer.add_scalar(
                        'Train/obj_loss',
                        obj_loss_meter.val, epoch)

                if self.config.gan:

                    if self.config.gan_feat_match_lambda:
                        self.writer.add_scalar(
                            'Train/g_loss_feat_match',
                            g_loss_feat_match_meter.val, epoch)

                    if self.config.ACGAN:
                        self.writer.add_scalar(
                            'Train/cluster_loss_real',
                            d_loss_cluster_real_meter.val, epoch)
                        self.writer.add_scalar(
                            'Train/cluster_loss_fake',
                            d_loss_cluster_fake_meter.val, epoch)

                    self.writer.add_scalar(
                        'Train/g_loss_fake',
                        g_loss_fake_meter.val, epoch)

                    self.writer.add_scalar(
                        'Train/d_loss_fake',
                        d_loss_fake_meter.val, epoch)
                    self.writer.add_scalar(
                        'Train/d_loss_real',
                        d_loss_real_meter.val, epoch)

                    self.writer.add_scalar(
                        'Train/d_fake',
                        d_fake_meter.val, epoch)
                    self.writer.add_scalar(
                        'Train/d_real',
                        d_real_meter.val, epoch)

                    self.writer.add_scalar(
                        'Train/d_loss',
                        d_loss_meter.val, epoch)
                    self.writer.add_scalar(
                        'Train/g_loss',
                        d_loss_meter.val, epoch)


            #############################
            # Validation & Save Model
            #############################
            if self.config.gpu == 0:
                self.validation(epoch)
            if self.config.gpu == 0:
                if epoch > 0 and epoch % 5 == 0:
                    self.save(self.G, 'G', epoch)
                    self.save(self.D, 'D', epoch)

    def validation(self, epoch):

        if self.config.gpu == 0:
            recon_loss_meter = LossMeter(len(self.val_loader))
            feat_loss_meter = LossMeter(len(self.val_loader))
            obj_loss_meter = LossMeter(len(self.val_loader))

            g_loss_fake_meter = LossMeter(len(self.val_loader))
            g_loss_feat_match_meter = LossMeter(len(self.val_loader))

            g_loss_cluster_meter = LossMeter(len(self.val_loader))

            d_loss_fake_meter = LossMeter(len(self.val_loader))
            d_loss_real_meter = LossMeter(len(self.val_loader))

            d_loss_cluster_fake_meter = LossMeter(len(self.val_loader))
            d_loss_cluster_real_meter = LossMeter(len(self.val_loader))

            d_fake_meter = LossMeter(len(self.val_loader))
            d_real_meter = LossMeter(len(self.val_loader))

            d_loss_meter = LossMeter(len(self.val_loader))
            g_loss_meter = LossMeter(len(self.val_loader))


        total_n_images = 0
        for step_i, data in enumerate(self.val_loader):

            # B = len(data[0])
            B = len(data['img'])
            total_n_images += B

            self.G.eval()
            if self.config.classifier:
                self.E.eval()
            if self.config.gan:
                self.D.eval()
            with torch.no_grad():
                out_dict = self.forward(data, train=False)

            if self.config.gpu == 0:

                if self.config.recon_loss_lambda:
                    recon_loss_meter.update(out_dict['recon_loss'])

                if self.config.feat_loss_lambda:
                    feat_loss_meter.update(out_dict['feat_loss'])
                if self.config.obj_loss_lambda:
                    obj_loss_meter.update(out_dict['obj_loss'])

                if self.config.gan:

                    if self.config.gan_feat_match_lambda:
                        g_loss_feat_match_meter.update(
                            out_dict['g_loss_feat_match'])
                    g_loss_fake_meter.update(out_dict['g_loss_fake'])

                    d_loss_fake_meter.update(out_dict['d_loss_fake'])
                    d_loss_real_meter.update(out_dict['d_loss_real'])

                    d_real_meter.update(out_dict['d_real'])
                    d_fake_meter.update(out_dict['d_fake'])

                    d_loss_meter.update(out_dict['d_loss'])
                    g_loss_meter.update(out_dict['g_loss'])

                    if self.config.ACGAN:
                        g_loss_cluster_meter.update(out_dict['g_loss_cluster'])
                        d_loss_cluster_real_meter.update(out_dict['d_loss_cluster_real'])
                        d_loss_cluster_fake_meter.update(out_dict['d_loss_cluster_fake'])

        if self.config.gpu == 0:
            log_str = f'Valid Epoch {epoch}'
            log_str += f' | Total # images: {total_n_images} |'
            if self.config.gan:
                log_str += f' (G) fake: {g_loss_fake_meter.val:.3f}'
                if self.config.gan_feat_match_lambda:
                    log_str += f' feat_match: {g_loss_feat_match_meter.val:.3f}'
                if self.config.ACGAN:
                    log_str += f' clst_fake: {d_loss_cluster_fake_meter.val:.3f}'
                    log_str += f' clst_real: {d_loss_cluster_real_meter.val:.3f}'
                log_str += f' (D) fake: {d_loss_fake_meter.val:.3f} real: {d_loss_real_meter.val:.3f}'
                log_str += f' | D(fake) {d_fake_meter.val:.3f} D(real) {d_real_meter.val:.3f}'
                log_str += f' | G loss: {g_loss_meter.val:.3f} D loss {d_loss_meter.val:.3f}'
            log_str += f' | '
            if self.config.recon_loss_lambda:
                log_str += f' recon loss: {recon_loss_meter.val:.3f}'
            if self.config.feat_loss_lambda:
                log_str += f' feat loss: {feat_loss_meter.val:.3f}'
            if self.config.obj_loss_lambda:
                log_str += f' obj loss: {obj_loss_meter.val:.3f}'


            print('\n' + log_str + '\n')
            self.logger.info(log_str)

            if self.config.recon_loss_lambda:
                self.writer.add_scalar(
                    'Valid/recon_loss',
                    recon_loss_meter.val, epoch)
            if self.config.feat_loss_lambda:
                self.writer.add_scalar(
                    'Valid/feat_loss',
                    feat_loss_meter.val, epoch)
            if self.config.obj_loss_lambda:
                self.writer.add_scalar(
                    'Valid/obj_loss',
                    obj_loss_meter.val, epoch)

            if self.config.gan:

                if self.config.gan_feat_match_lambda:
                    self.writer.add_scalar(
                        'Valid/g_loss_feat_match',
                        g_loss_feat_match_meter.val, epoch)

                if self.config.ACGAN:
                    self.writer.add_scalar(
                        'Valid/cluster_loss_real',
                        d_loss_cluster_real_meter.val, epoch)
                    self.writer.add_scalar(
                        'Valid/cluster_loss_fake',
                        d_loss_cluster_fake_meter.val, epoch)

                self.writer.add_scalar(
                    'Valid/g_loss_fake',
                    g_loss_fake_meter.val, epoch)

                self.writer.add_scalar(
                    'Valid/d_loss_fake',
                    d_loss_fake_meter.val, epoch)
                self.writer.add_scalar(
                    'Valid/d_loss_real',
                    d_loss_real_meter.val, epoch)

                self.writer.add_scalar(
                    'Valid/d_fake',
                    d_fake_meter.val, epoch)
                self.writer.add_scalar(
                    'Valid/d_real',
                    d_real_meter.val, epoch)

                self.writer.add_scalar(
                    'Valid/d_loss',
                    d_loss_meter.val, epoch)
                self.writer.add_scalar(
                    'Valid/g_loss',
                    d_loss_meter.val, epoch)

    def sample(self, epoch, step_i, from_feat=False):
        with torch.no_grad():
            real_img = self.debug_real_img
            B = real_img.size(0)

            if from_feat:
                img_grids_feat = self.debug_img_grids_feat
                img_grids_feat = img_grids_feat.cuda().view(B, self.n_grid, self.n_grid, -1)
                img_grids_feat = img_grids_feat.permute(0,3,1,2)
            else:
                if self.config.clustering:
                    cluster_ids = self.debug_cluster_ids.cuda().view(
                        B * self.n_grid * self.n_grid)
                    img_grids_feat = self.Emb(cluster_ids)
                    img_grids_feat = img_grids_feat.view(B, self.n_grid, self.n_grid, -1)
                    img_grids_feat = img_grids_feat.permute(0,3,1,2)

                elif self.config.independent_grid:
                    img_grids = grid_view(real_img, self.config.n_grid)
                    # [B x n_grid x n_grid, feat_dim]
                    _, img_grids_feat, _ = self.E(self.imagenet_norm(img_grids),
                                                obj_class=False, pool=True)
                    img_grids_feat = img_grids_feat.view(B, self.n_grid, self.n_grid, -1)
                    img_grids_feat = img_grids_feat.permute(0,3,1,2)
                else:
                    downsample_ratio = self.config.resize_input_size // self.config.n_grid

                    img_grids_feat = self.E(self.imagenet_norm(real_img),
                                            last_feat_only=True,
                                            upscale_ratio=downsample_ratio)

            # [B, feat_dim, n_grid, n_grid]
            assert img_grids_feat.size(0) == B
            assert img_grids_feat.size()[2:] == (self.n_grid, self.n_grid)

            # [B, 3, resize_target_size, resize_target_size]
            fake_img = self.G(img_grids_feat)

            fake_img = self.denorm(fake_img)  # [-1, 1] => [0, 1]

            assert fake_img.size() == (B, self.config.n_channel,
                                       self.config.resize_target_size,
                                       self.config.resize_target_size), fake_img.size()

            if from_feat:
                fake_img_path = self.img_log_dir.joinpath(
                    f"fake_epoch{epoch:02d}_step{step_i:04d}_feat.png")
            else:
                fake_img_path = self.img_log_dir.joinpath(
                    f"fake_epoch{epoch:02d}_step{step_i:04d}.png")

            self.pbar.write(f'Save fake images at {fake_img_path}')
            save_image(fake_img, fake_img_path)

            real_img_path = self.img_log_dir.joinpath("real.png")
            if not real_img_path.exists():
                self.pbar.write(f'Save real images at {real_img_path}')
                # real_img = self.imagenet_denorm(real_img)
                save_image(real_img, real_img_path)

    def save(self, model, name, epoch):
        torch.save(model.state_dict(),
                   self.ckpt_dir.joinpath(f"{name}_{epoch}.pth"))
