# coding=utf-8
# Copyleft 2019 project LXRT.

import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import collections
from pathlib import Path

import sys
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import DataParallel
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from PIL import Image
# from torch.utils.data.dataloader import DataLoader
import logging
import shutil
from copy import deepcopy

# from apex import amp

# from param import args
from param import parse_args
from tasks.imggen_model import ImggenModel
# from tasks.imggen_layers import Generator, Discriminator, ResNetEncoder

from tasks.imggen_data import get_loader, box_position

# image generator path
image_generator_dir = Path(__file__).resolve().parent.parent.parent.parent.joinpath('image_generator')
# print(image_generator_dir)

def is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')

def reduce_dict(_results, gpu=0, reduce_gpu=0):
    results = {}
    for k, v in _results.items():
        if type(v) == torch.Tensor:
            results[k] = v.cuda(gpu).float()
        else:
            results[k] = torch.tensor(v).cuda(gpu).float()

    names = []
    values = []
    for k in sorted(results.keys()):
        names.append(k)
        values.append(results[k])
    values = torch.stack(values, dim=0)
    dist.reduce(values, dst=reduce_gpu)

    if gpu == 0:
        reduced_dict = {}
        for k, v in zip(names, values):
            reduced_dict[k] = v.item()
        return reduced_dict
    else:
        return None


def load_state_dict(state_dict_path, loc='cpu'):
    state_dict = torch.load(state_dict_path, map_location=loc)
    # Change Multi GPU to single GPU
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            new_state_dict[key[len("module."):]] = value
    return new_state_dict


class ImgGen:
    def __init__(self, args, train_loader=None, val_loader=None, logger=None, train=True):
        self.args = args
        self.max_text_length = args.max_text_length

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.logger = logger

        self.verbose = True
        if self.args.distributed:
            if self.args.gpu != 0:
                self.verbose = False

        # Build LXRT Model
        from lxrt.entry import set_visual_config
        set_visual_config(args)
        self.model = ImggenModel.from_pretrained(
            "bert-base-uncased",
            args=args,
        )

        # Load pre-trained weights
        if args.load is not None:
            lxmert_ckpt = args.load +'_LXRT.pth'
            state_dict = load_state_dict(lxmert_ckpt, 'cpu')
            results = self.model.load_state_dict(state_dict, strict=False)
            if self.verbose:
                print('LXRT loaded from ', lxmert_ckpt)
                print(results)

        if args.resize_input_size // args.grid_size == 8:
            out_layer = 'conv3'
        elif args.resize_input_size // args.grid_size == 16:
            out_layer = 'conv4'
        elif args.resize_input_size // args.grid_size == 32:
            out_layer = 'conv5'

        if train:
            self.resnet = ResNetEncoder('resnet50', out=out_layer).eval()
            self.resnet.requires_grad_(False)

        emb_dim = 2048
        code_dim = 256
        norm_type = 'spade_in'
        SN = True
        ACGAN = True
        CGAN = True


        assert self.args.grid_size == 8

        sys.path.append(str(image_generator_dir))
        from layers import Generator, Discriminator, ResNetEncoder

        if self.args.resize_target_size == 64:
            base_dim = 64
            g_ckpt_path = image_generator_dir.joinpath('pretrained/G.pth')
            # d_ckpt_path = '/home/jaeminc/Dropbox/Projects/AI2/ResNet_AE/logs/mscoco/ResNetGenerator/May16_15-00_GPU4_8_64/ckpt/D_50.pth'

        elif self.args.resize_target_size == 128:
            base_dim = 64
            g_ckpt_path = '/home/jaeminc/Dropbox/Projects/AI2/ResNet_AE/logs/mscoco/ResNetGenerator/May16_18-29_GPU8_8_128/ckpt/G_50.pth'
            # d_ckpt_path = '/home/jaeminc/Dropbox/Projects/AI2/ResNet_AE/logs/mscoco/ResNetGenerator/May16_18-29_GPU8_8_128/ckpt/D_50.pth'

        elif self.args.resize_target_size == 256:
            base_dim = 32
            g_ckpt_path = image_generator_dir.joinpath('pretrained/G_60.pth')

            # g_ckpt_path = '/home/jaeminc/Dropbox/Projects/AI2/ResNet_AE/logs/mscoco/ResNetGenerator/May20_22-06_GPU8_8_256_epoch101/ckpt/G_60.pth'

            # g_ckpt_path = '/home/jaeminc/Dropbox/Projects/AI2/ResNet_AE/logs/mscoco/ResNetGenerator/May16_22-55_GPU8_8_256/ckpt/G_50.pth'

            # g_ckpt_path = '/home/jaeminc/Dropbox/Projects/AI2/ResNet_AE/logs/mscoco/ResNetGenerator/May20_22-06_GPU8_8_256_epoch101/ckpt/G_100.pth'
            # g_ckpt_path = '/home/jaeminc/Dropbox/Projects/AI2/ResNet_AE/logs/mscoco/ResNetGenerator/May23_18-30_GPU6_8_256_Proj_origfeat/ckpt/G_50.pth'

        G = Generator(
            base_dim=base_dim,
            emb_dim=emb_dim,
            norm_type=norm_type,
            target_size=self.args.resize_target_size,
            init_H=self.args.grid_size,
            init_W=self.args.grid_size,
            SN=SN,
            codebook_dim=code_dim,
        )

        # ckpt_path = '/home/jaeminc/Dropbox/Projects/AI2/ResNet_AE/logs/mscoco/ResNetGenerator/Mar11_23-05_GPU4/ckpt/G_100.pth'
        # ckpt_path = '/home/jaeminc/Dropbox/Projects/AI2/ResNet_AE/logs/mscoco/ResNetGenerator/Mar17_19-30_GPU8/ckpt/G_40.pth'
        # ckpt_path = '/home/jaeminc/Dropbox/Projects/AI2/ResNet_AE/logs/mscoco/ResNetGenerator/Apr04_18-25_GPU8/ckpt/G_50.pth'
        g_state_dict = load_state_dict(g_ckpt_path, 'cpu')
        results = G.load_state_dict(g_state_dict, strict=False)
        if self.verbose:
            print('G loaded from', g_ckpt_path)
            print(results)
        G.eval()

        self.model.G = G

        if train and args.gan:
            self.D = Discriminator(
                base_dim=base_dim,
                emb_dim=emb_dim,
                target_size=self.args.resize_target_size,
                init_H=self.args.grid_size,
                init_W=self.args.grid_size,
                ACGAN=ACGAN,
                SN=SN,
                CGAN=CGAN
            )
            self.D.eval()

            if self.args.D_tune:
                for i, resblock in enumerate(self.D.resblocks):
                    if i < self.args.D_freeze_layers:
                        resblock.requires_grad_(False)
            # else:
            #     self.D.requires_grad_(False)

            d_ckpt_path = '/home/jaeminc/Dropbox/Projects/AI2/ResNet_AE/logs/mscoco/ResNetGenerator/Apr04_18-25_GPU8/ckpt/D_50.pth'
            d_state_dict = load_state_dict(d_ckpt_path, 'cpu')
            results = self.D.load_state_dict(d_state_dict, strict=False)
            if self.verbose:
                print('D loaded from', d_ckpt_path)
                print(results)

        if args.distributed:
            if 'bn' in self.args.g_norm_type:
                # G = apex.parallel.convert_syncbn_model(G)
                self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)

        if self.args.clustering:
            pass
        else:
            self.resnet_to_code = nn.Sequential(
                nn.Conv2d(self.args.feat_dim, self.args.codebook_dim, 1, padding=0),
                nn.Tanh()
            )
            self.model.resnet_to_code = self.resnet_to_code

        # GPU Options
        print(f'Model Launching at GPU {self.args.gpu}')
        from time import time
        start = time()
        self.model.cuda(args.gpu)
        if train:
            self.resnet.cuda(args.gpu)

            if self.args.gan:
                self.D.cuda(args.gpu)

        # Losses and optimizer
        if train:
            params = [
                # {'params': self.model.lxrt_encoder.parameters()},
                # {'params': self.model.vis_encoding.parameters()},
                # {'params': list(self.model.cls.seq_relationship.parameters()) + list(self.model.cls.predictions.transform.parameters()) + [self.model.cls.predictions.bias]},
                # {'params': list(self.model.code_head[:-1].parameters()) + [self.model.code_head[-1].bias]},
                # {'params': self.model.Emb.parameters(), 'lr': 1e-6},

                {'params': self.model.bert.parameters()},
                {'params': list(self.model.cls.seq_relationship.parameters()) + list(self.model.cls.predictions.transform.parameters()) + [self.model.cls.predictions.bias]},
                {'params': self.model.mask_feat},
                {'params': list(self.model.obj_predict_head.transform.parameters()) + list(self.model.obj_predict_head.linear_feat.parameters())},

                {'params': self.model.vis_emb.parameters(),
                    # 'lr': 1e-5
                },
            ]

            if self.args.optim == 'bert':
                params += [{'params': self.model.G.parameters(), 'lr': 4e-4, 'b1': 0.0, 'b2': 0.999, 'e': 1e-7}]
            elif 'adam' in self.args.optim:
                params += [{'params': self.model.G.parameters(), 'lr': 4e-4, 'betas': [0.0, 0.999], 'eps': 1e-7}]

            if 'bert' in args.optim:
                batch_per_epoch = len(train_loader)
                t_total = int(batch_per_epoch * args.epochs)
                warmup_ratio = 0.1
                warmup_iters = int(t_total * warmup_ratio)
                if self.verbose:
                    print("Batch per epoch: %d" % batch_per_epoch)
                    print("Total Iters: %d" % t_total)
                    print('Warmup ratio:', warmup_ratio)
                    print("Warm up Iters: %d" % warmup_iters)
                from lxrt.optimization import BertAdam

                self.optim = BertAdam(params,
                                      lr=args.lr,
                                      warmup=warmup_ratio,
                                      t_total=t_total,
                                      # e=1e-4
                                      )

            else:
                self.optim = args.optimizer(params, args.lr)

            # self.g_optim = optim.Adam(self.G.parameters(),
            #                           lr=0.0004, betas=[0.0, 0.999])

            if self.args.gan:
                if self.args.D_tune:
                    self.d_optim = args.optimizer(filter(lambda p: p.requires_grad, self.D.parameters()),
                                              lr=0.0001, betas=[0.0, 0.999], eps=1e-7)

            if args.mixed_precision:
                self.model, self.optim = amp.initialize(
                    self.model, self.optim, opt_level='O1', verbosity=self.verbose)
                # [self.model, self.D], [self.optim, self.d_optim] = amp.initialize(
                #     [self.model, self.D], [self.optim, self.d_optim],
                #     opt_level='O1', verbosity=self.verbose)

                # self.G, self.g_optim = amp.initialize(
                #     self.G, self.g_optim, opt_level='O1', verbosity=self.verbose)

                if self.args.gan:
                    if self.args.D_tune:
                        self.D, self.d_optim = amp.initialize(
                            self.D, self.d_optim, opt_level='O1', verbosity=self.verbose)

        if args.multiGPU:
            if args.distributed:
                self.model = DDP(self.model, device_ids=[args.gpu],
                                 find_unused_parameters=True,
                                 )

                # self.G = DDP(self.G, device_ids=[args.gpu],
                #              find_unused_parameters=True,
                #              )

                if self.args.gan:
                    if self.args.D_tune:
                        self.D = DDP(self.D, device_ids=[args.gpu],
                                     find_unused_parameters=True
                                     )
            else:
                self.model = nn.DataParallel(self.model)
                self.D = nn.DataParallel(self.D)

        if args.gpu == 0:
            print(f'It took {time() - start:.1f}s')

            # Checkpointing & Image logging
            if train:
                self.output = args.output
                os.makedirs(self.output, exist_ok=True)

                self.img_log_dir = self.args.log_dir.joinpath('images')
                self.img_log_dir.mkdir(exist_ok=True)

                batch_entry = next(iter(self.val_loader))
                self.debug_batch = deepcopy(batch_entry)
                print('Sample from initilization')
                self.sample(self.debug_batch, -1, -1, tensorboard=False)

        if args.distributed:
            dist.barrier()


    def encode(self, batch_entry, task='vis_mask'):
        if task == 'word_mask':
            word_id = batch_entry['masked_word_id'].cuda(self.args.gpu)
        elif task == 'matched':
            word_id = batch_entry['other_word_id'].cuda(self.args.gpu)
        elif task == 'vis_mask':
            word_id = batch_entry['word_id'].cuda(self.args.gpu)

        word_attention_mask = word_id > 0
        token_type_ids = torch.zeros_like(word_id)

        grid = batch_entry['boxes'].cuda(self.args.gpu)
        vis_mask = batch_entry['vis_mask'].cuda(self.args.gpu)

        B = len(word_id)

        if self.args.clustering:
            # [B, n_grids]
            cluster_id = batch_entry['cluster_id'].cuda(self.args.gpu)

            # [B, n_grids, code_dim]
            if type(self.model) in [DDP, DataParallel]:
                code = self.model.module.vis_emb(cluster_id)
            else:
                code = self.model.vis_emb(cluster_id)

            resnet_feat = None
        else:
            # [B, feat_dim, grid_size, grid_size]
            with torch.no_grad():
                real_img = batch_entry['img'].cuda(self.args.gpu)
                resnet_feat = self.resnet(self.imagenet_norm(real_img))
            # [B, code_dim, grid_size, grid_size]
            if type(self.model) in [DDP, DataParallel]:
                code = self.model.module.resnet_to_code(resnet_feat)
            else:
                code = self.model.resnet_to_code(resnet_feat)

            if self.args.code_regression:
                # [B, n_grids, code_dim]
                code = code.permute(0,2,3,1).view(B, self.args.n_grids, self.args.codebook_dim)
            else:
                # quantization
                if type(self.model) in [DDP, DataParallel]:
                    code, code_id = self.model.module.quantize_code(code, weight_sg=False)
                else:
                    code, code_id = self.model.quantize_code(code, weight_sg=False)
                # [B, n_grids, code_dim]
                code = code.permute(0,2,3,1).view(B, self.args.n_grids, self.args.codebook_dim)

        visn_feats = (
            code,
            grid,
            vis_mask
        )

        sent_feats = (
            word_id,
            token_type_ids,
            word_attention_mask,
        )
        return visn_feats, sent_feats, resnet_feat

    def forward(self, batch_entry, task='vis_mask', G_input_feed=False):
        results = {}

        visn_feats, sent_feats, resnet_feat = self.encode(batch_entry)
        code, grid, vis_mask = visn_feats
        word_id, token_type_ids, word_attention_mask = sent_feats
        B = code.size(0)

        label_dict = {}
        if task == 'word_mask':
            word_labels = batch_entry['word_labels'].cuda(self.args.gpu)
            label_dict['word_labels'] = word_labels
        elif task == 'vis_mask':
            # if not self.args.code_regression:
            #     code_id = code_id.detach().view(B, self.args.n_grids)
            #     if not self.args.VMP_smart:
            #         code_id[~vis_mask] = -1
            #     label_dict['code_labels'] = code_id
            if self.args.clustering:
                if not G_input_feed:
                    label_dict['input_code_id'] = batch_entry['cluster_id'].clone().detach().cuda(self.args.gpu)
                code_label = batch_entry['cluster_id'].clone().detach().cuda(self.args.gpu)
                if not self.args.VMP_smart:
                    code_label[~vis_mask] = -1
                label_dict['code_labels'] = code_label
        elif task == 'matched':
            matched_label = batch_entry['matched_label'].cuda(self.args.gpu)#)
            label_dict['matched_labels'] = matched_label

        if task == 'vis_mask' and self.args.VMP_smart:
            with torch.no_grad():
                self.model.eval()
                out_dict = self.model(visn_feats, sent_feats, label_dict, task, calc_loss=False)
                # [B, n_grids, code_dim]
                pred_code = out_dict['pred_code']

                code = torch.where(vis_mask.view(B, self.args.n_grids, 1).bool(),
                                   pred_code,
                                   code)
                vis_mask_2 = batch_entry['vis_mask_2'].cuda(self.args.gpu)
                visn_feats = (
                    code,
                    grid,
                    vis_mask_2
                )
                code_label = batch_entry['cluster_id'].clone().detach().cuda(self.args.gpu)
                code_label[~vis_mask_2] = -1
                label_dict['code_labels'] = code_label

            self.model.train()
            out_dict = self.model(visn_feats, sent_feats, label_dict, task, calc_loss=True, G_input_feed=G_input_feed)
        else:
            out_dict = self.model(visn_feats, sent_feats, label_dict, task, calc_loss=True, G_input_feed=G_input_feed)

        if 'vis_mask' == task:

            real_img = batch_entry['img'].cuda(self.args.gpu)
            resized_target_img = F.interpolate(
                real_img,
                size=(self.args.resize_target_size, self.args.resize_target_size),
                mode='bilinear', align_corners=False)

            if self.args.clustering:
                code_loss = out_dict['code_loss']

                fake_img = out_dict['fake_img']

                # [B, feat_dim, H, W]
                fake_resnet_feat = self.resnet(self.imagenet_norm(self.denorm(fake_img)), last_feat=True)

                with torch.no_grad():
                    target_resnet_feat = self.resnet(self.imagenet_norm(real_img))
                    resized_target_resnet_feat = self.resnet(self.imagenet_norm(resized_target_img), last_feat=True)

                perceptual_loss = F.smooth_l1_loss(fake_resnet_feat, resized_target_resnet_feat.detach())
                # perceptual_loss = 0

                if self.args.pixel_loss_lambda:
                    pixel_loss = F.smooth_l1_loss(fake_img, self.norm(resized_target_img.detach()))
                else:
                    pixel_loss = torch.zeros_like(code_loss)

                vis_loss = perceptual_loss * self.args.perceptual_loss_lambda + pixel_loss + code_loss

                if self.args.gan:
                    D_fake, D_layers_fake = self.D(fake_img, target_resnet_feat, output_layers=True)

                    if self.args.hinge:
                        g_loss_fake = -D_fake.mean()
                    else:
                        g_loss_fake = -F.logsigmoid(D_fake).mean()

                    if self.args.gan_feat_match_lambda:
                        D_real, D_layers_real = self.D(self.norm(resized_target_img), target_resnet_feat,  output_layers=True)

                        g_loss_feat_match = torch.zeros_like(pixel_loss)
                        for recon_layer, tgt_layer in zip(
                                D_layers_fake[:self.args.gan_feat_match_layers],
                                D_layers_real[:self.args.gan_feat_match_layers]):
                            g_loss_feat_match = g_loss_feat_match + F.smooth_l1_loss(
                                recon_layer, tgt_layer)

                    vis_loss += g_loss_fake * self.args.gan_loss_lambda + g_loss_feat_match * self.args.gan_feat_match_lambda

                    results['g_loss_fake'] = g_loss_fake
                    results['g_loss_feat_match'] = g_loss_feat_match

                results['perceptual_loss'] = perceptual_loss
                results['pixel_loss'] = pixel_loss
                results['code_loss'] = code_loss

                results['vis_loss'] = vis_loss

                if self.args.gan:
                    results['fake_img'] = fake_img.clone().detach()
                    results['real_img'] = real_img.clone().detach()
                    results['resized_target_img'] = resized_target_img.clone().detach()
                    results['target_resnet_feat'] = target_resnet_feat.clone().detach()

        elif 'word_mask' == task:
            results['lm_loss'] = out_dict['lm_loss']
        elif 'matched' == task:
            results['matched_loss'] = out_dict['matched_loss']

        return results

    def train(self):
        LOSSES_NAME = self.args.LOSSES_NAME
        task_dict = {
            'Mask_LM': 'word_mask',
            'Matched': 'matched',
            # 'Mask_Vis': 'vis_mask',
            'Mask_Code': 'vis_mask',
            'Perceptual': 'vis_mask',
            'Recon': 'vis_mask',

            'D_feat_match': 'vis_mask',
            'D(fake)': 'vis_mask',
            'D(real)': 'vis_mask',
            'D loss': 'vis_mask',
            'G loss': 'vis_mask'
        }

        args = self.args
        if self.verbose:
            loss_meter = LossMeter()
            loss_meters = [LossMeter() for _ in range(len(LOSSES_NAME))]

            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=self.args.log_dir)

        dist.barrier()

        self.optim.zero_grad()
        # self.d_optim.zero_grad()

        if self.args.dry:
            # Validation
            results = self.evaluate_epoch(self.val_loader, split=args.valid)

        n_update = 0
        for epoch in range(self.args.epochs):
            self.model.train()
            loss_counts = [0 for _ in range(len(LOSSES_NAME))]

            if self.args.distributed:
                self.train_loader.sampler.set_epoch(epoch)
            if self.verbose:
                pbar = tqdm(total=len(self.train_loader), ncols=200)

            epoch_results = {
                'lm_loss': 0,
                'lm_loss_count': 0,
                'vis_loss': 0,
                'vis_loss_count': 0,
                'matched_loss': 0,
                'matched_loss_count': 0,

                'perceptual_loss': 0,
                'pixel_loss': 0,
                'code_loss': 0,

                'g_loss_fake': 0,
                'g_loss_feat_match': 0,

                'd_loss_fake': 0,
                'd_loss_real': 0,
                'd_loss': 0,
                'D(real)': 0,
                'D(fake)': 0,
            }

            # vis_mask_sampled = False

            for step_i, batch in enumerate(self.train_loader):
                # task = random.choice(self.args.MASK_MODALITY)
                task_i = step_i % len(self.args.MASK_MODALITY)
                task = self.args.MASK_MODALITY[task_i]

                # with torch.autograd.set_detect_anomaly(True):

                G_input_feed = False
                if self.args.G_scheduled_sampling:
                    G_sample_ratio = (epoch+1) / 10
                    if random.random() < G_sample_ratio:
                        G_input_feed = True

                results = self.forward(batch, task, G_input_feed=G_input_feed)

                if task == 'vis_mask':
                    epoch_results['vis_loss_count'] += 1
                    loss = results['vis_loss']
                elif task == 'word_mask':
                    epoch_results['lm_loss_count'] += 1
                    loss = results['lm_loss']
                elif task == 'matched':
                    epoch_results['matched_loss_count'] += 1
                    loss = results['matched_loss']

                update = True
                if self.args.update_freq > 1:
                    if step_i == 0:
                        update = False
                    elif step_i % self.args.update_freq == 0 or step_i == len(self.train_loader) - 1:
                        update = True
                    else:
                        update = False

                if self.args.mixed_precision:
                    with amp.scale_loss(loss, self.optim) as scaled_loss:

                        scaled_loss.backward()

                    if update:
                        nn.utils.clip_grad_norm_(amp.master_params(self.optim), 1.)
                else:
                    loss.backward()
                    if update:
                        nn.utils.clip_grad_norm_(self.model.parameters(), 1.)

                if update:
                    self.optim.step()
                    self.optim.zero_grad()

                #========= D Update ===========#
                if task == 'vis_mask' and self.args.gan:
                    self.model.eval()
                    if self.args.D_tune:
                        self.D.train()

                    fake_img = results['fake_img'].detach().clone()
                    resized_target_img = results['resized_target_img'].detach().clone()
                    target_resnet_feat = results['target_resnet_feat'].detach().clone()

                    D_real = self.D(self.norm(resized_target_img), target_resnet_feat, False)
                    D_fake = self.D(fake_img, target_resnet_feat, False)

                    if self.args.hinge:
                        d_loss_real = F.relu(1.0 - D_real).mean()
                        d_loss_fake = F.relu(1.0 + D_fake).mean()
                    else:
                        d_loss_real = -F.logsigmoid(D_real).mean()
                        # d_loss_fake = F.logsigmoid(D_fake).mean()
                        d_loss_fake = -(1 - torch.sigmoid(D_fake)).log().mean()

                    d_loss = d_loss_real + d_loss_fake

                    if self.args.D_tune:
                        self.d_optim.zero_grad()
                        if self.args.mixed_precision:
                            with amp.scale_loss(d_loss, self.d_optim) as d_scaled_loss:
                                d_scaled_loss.backward()
                            torch.nn.utils.clip_grad_norm_(
                                amp.master_params(self.d_optim), 1.)
                        else:
                            d_loss.backward()
                            torch.nn.utils.clip_grad_norm_(
                                self.D.parameters(), 1.)
                        self.d_optim.step()

                    results['D(real)'] = D_real.mean()
                    results['D(fake)'] = D_fake.mean()
                    results['d_loss_real'] = d_loss_real
                    results['d_loss_fake'] = d_loss_fake
                    results['d_loss'] = d_loss

                    self.model.train()
                    self.D.eval()
                #==============================#

                for k, v in results.items():
                    try:
                        if ('img' not in k ) and ('resnet_feat' not in k):
                            epoch_results[k] += v.item()
                    except ValueError:
                        print(k, v)
                        exit()

                if 'bert' in self.args.optim:
                    try:
                        lr = self.optim.lr
                    except AttributeError:
                        lr = 0.
                else:
                    lr = self.args.lr

                if self.verbose:

                    desc_str = f'Epoch {epoch} | LR {lr:.6f} | '
                    if self.args.G_scheduled_sampling:
                        desc_str += f'SS_ratio:{G_sample_ratio:.2f} | '
                    if self.args.word_mask_predict:
                        desc_str += f'Word Mask: Uniform (MP) | '
                    elif self.args.word_mask_rate > 0:
                        desc_str += f'Word Mask: {self.args.word_mask_rate:.2f} | '

                    if self.args.vis_mask_predict:
                        desc_str += f'Vis Mask: Uniform (MP) |'
                    else:
                        desc_str += f'Vis Mask: {self.args.obj_mask_rate:.2f} |'

                    for i, (loss_name, loss_meter) in enumerate(zip(LOSSES_NAME, loss_meters)):
                        if task_dict[loss_name] == task:
                            if task == 'vis_mask':
                                if loss_name == 'Mask_Code':
                                    loss_meter.update(results['code_loss'].item())
                                elif loss_name == 'Perceptual':
                                    loss_meter.update(results['perceptual_loss'].item())
                                elif loss_name == 'Recon':
                                    loss_meter.update(results['pixel_loss'].item())

                                if self.args.gan:
                                    if loss_name == 'D_feat_match':
                                        loss_meter.update(results['g_loss_feat_match'].item())
                                    elif loss_name == 'D(fake)':
                                        loss_meter.update(results['D(fake)'].item())
                                    elif loss_name == 'D(real)':
                                        loss_meter.update(results['D(real)'].item())
                                    elif loss_name == 'D loss':
                                        loss_meter.update(results['d_loss'].item())
                                    elif loss_name == 'G loss':
                                        loss_meter.update(results['g_loss_fake'].item())

                            elif task == 'word_mask':
                                loss_meter.update(results['lm_loss'])
                            elif task == 'matched':
                                loss_meter.update(results['matched_loss'])
                            loss_counts[i] += 1
                        if len(loss_meter) > 0:
                            loss_count = loss_counts[i]
                            if loss_name in ['Mask_LM', 'Matched', 'Mask_Code']:
                                desc_str += f' {loss_name} ({loss_count}) {loss_meter.val:.2f}'
                            else:
                                desc_str += f' {loss_name} {loss_meter.val:.2f}'

                            if step_i % 10 == 0:
                                global_step = len(self.train_loader) * epoch + step_i
                                self.writer.add_scalar(f'Train_steps/{loss_name}', loss_meter.val, global_step)

                    if update:
                        n_update += 1
                    desc_str += f' | Total Update: {n_update}'

                    pbar.set_description(desc_str)

                    pbar.update(1)

                    img_log_freq = len(self.train_loader) // 10
                    if (step_i % img_log_freq == 0) or (step_i + 1 == len(self.train_loader)):
                        self.model.eval()
                        self.sample(self.debug_batch, epoch, step_i, pbar=pbar, tensorboard=True)
                        self.model.train()

                dist.barrier()

            if self.verbose:
                pbar.close()

            results = reduce_dict(epoch_results, self.args.gpu)
            if self.args.gpu == 0:

                total_loss = results['lm_loss'] + results['vis_loss'] + results['matched_loss']
                total_count = results['lm_loss_count'] + results['vis_loss_count'] + results['matched_loss_count']

                avg_train_loss = total_loss / total_count
                losses_str = f"Train Loss: {avg_train_loss:.4f}\n"
                loss_names = ['lm_loss', 'vis_loss', 'matched_loss']
                vis_loss_names = ['perceptual_loss', 'pixel_loss', 'code_loss']
                if self.args.gan:
                    vis_loss_names += ['g_loss_fake', 'g_loss_feat_match']
                    vis_loss_names += ['D(fake)', 'D(real)']
                for i, name in enumerate(loss_names + vis_loss_names):
                    loss = results[name]
                    if name in vis_loss_names:
                        loss_count = int(results['vis_loss_count'])
                    else:
                        loss_count = int(results[name+'_count'])
                    avg_loss = loss/loss_count
                    losses_str += f"{name} ({loss_count}): {avg_loss:.2f} "
                    self.writer.add_scalar(f'Train/{name}', avg_loss, epoch)

                losses_str += '\n'
                print(losses_str)
                self.logger.info(losses_str)

            # Validation
            valid_results = self.evaluate_epoch(self.val_loader, epoch=epoch, split=args.valid)

            valid_results = reduce_dict(valid_results, self.args.gpu)
            if self.args.gpu == 0:

                valid_total_loss = valid_results['lm_loss'] + valid_results['vis_loss'] + valid_results['matched_loss']
                valid_total_count = valid_results['lm_loss_count'] + valid_results['vis_loss_count'] + valid_results['matched_loss_count']

                avg_valid_loss = valid_total_loss / valid_total_count
                losses_str = f"Valid Loss: {avg_valid_loss:.4f}\n"
                loss_names = ['lm_loss', 'vis_loss', 'matched_loss']
                vis_loss_names = ['perceptual_loss', 'pixel_loss', 'code_loss']
                if self.args.gan:
                    vis_loss_names += ['g_loss_fake', 'g_loss_feat_match']
                    vis_loss_names += ['D(fake)', 'D(real)']
                for i, name in enumerate(loss_names + vis_loss_names):
                    loss = valid_results[name]
                    if name in vis_loss_names:
                        loss_count = int(valid_results['vis_loss_count'])
                    else:
                        loss_count = int(valid_results[name+'_count'])
                    avg_loss = loss/loss_count
                    losses_str += f"{name} ({loss_count}): {avg_loss:.2f} "
                    self.writer.add_scalar(f'Valid/{name}', avg_loss, epoch)

                losses_str += '\n'
                print(losses_str)
                self.logger.info(losses_str)

                self.save("Epoch%02d" % (epoch + 1))

            dist.barrier()

        if self.verbose:
            self.save("LAST")

    def evaluate_epoch(self, loader, epoch=0, split='val'):
        LOSSES_NAME = self.args.LOSSES_NAME
        # task_dict = {
        #     'Mask_LM': 'word_mask',
        #     'Matched': 'matched',
        #     'Mask_Vis': 'vis_mask',
        # }
        task_dict = {
            'Mask_LM': 'word_mask',
            'Matched': 'matched',
            # 'Mask_Vis': 'vis_mask',
            'Mask_Code': 'vis_mask',
            'Perceptual': 'vis_mask',
            'Recon': 'vis_mask',

            'D_feat_match': 'vis_mask',
            'D(fake)': 'vis_mask',
            'D(real)': 'vis_mask',
            'D loss': 'vis_mask',
            'G loss': 'vis_mask'
        }

        epoch_results = {
            'lm_loss': 0,
            'lm_loss_count': 0,
            'vis_loss': 0,
            'vis_loss_count': 0,
            'matched_loss': 0,
            'matched_loss_count': 0,

            'perceptual_loss': 0,
            'pixel_loss': 0,
            'code_loss': 0,

            'g_loss_fake': 0,
            'g_loss_feat_match': 0,

            'd_loss_fake': 0,
            'd_loss_real': 0,
            'd_loss': 0,
            'D(real)': 0,
            'D(fake)': 0,
        }

        self.model.eval()
        with torch.no_grad():
            if self.verbose:
                loss_meter = LossMeter()
                loss_meters = [LossMeter() for _ in range(len(LOSSES_NAME))]

                loss_counts = [0 for _ in range(len(LOSSES_NAME))]

                pbar = tqdm(total=len(loader), ncols=180)

            for step_i, batch in enumerate(loader):
                # task = random.choice(self.args.MASK_MODALITY)
                task_i = step_i % len(self.args.MASK_MODALITY)
                task = self.args.MASK_MODALITY[task_i]

                results = self.forward(batch, task)

                if task == 'vis_mask':
                    epoch_results['vis_loss_count'] += 1
                elif task == 'word_mask':
                    epoch_results['lm_loss_count'] += 1
                elif task == 'matched':
                    epoch_results['matched_loss_count'] += 1

                #========= GAN Loss ===========#
                if task == 'vis_mask' and self.args.gan:
                    self.model.eval()
                    if self.args.D_tune:
                        self.D.eval()

                    fake_img = results['fake_img'].detach().clone()
                    resized_target_img = results['resized_target_img'].detach().clone()
                    target_resnet_feat = results['target_resnet_feat'].detach().clone()

                    D_real = self.D(self.norm(resized_target_img), target_resnet_feat, False)
                    D_fake = self.D(fake_img, target_resnet_feat, False)

                    if self.args.hinge:
                        d_loss_real = F.relu(1.0 - D_real).mean()
                        d_loss_fake = F.relu(1.0 + D_fake).mean()
                    else:
                        d_loss_real = -F.logsigmoid(D_real).mean()
                        # d_loss_fake = F.logsigmoid(D_fake).mean()
                        d_loss_fake = -(1 - torch.sigmoid(D_fake)).log().mean()

                    d_loss = d_loss_real + d_loss_fake
                    results['D(real)'] = D_real.mean()
                    results['D(fake)'] = D_fake.mean()
                    results['d_loss_real'] = d_loss_real
                    results['d_loss_fake'] = d_loss_fake
                    results['d_loss'] = d_loss

                #==============================#

                for k, v in results.items():
                    try:
                        if ('img' not in k ) and ('resnet_feat' not in k):
                            epoch_results[k] += v.item()
                    except ValueError:
                        print(k, v)
                        exit()

                if self.verbose:
                    desc_str = f'Valid Epoch {epoch} | '

                    for i, (loss_name, loss_meter) in enumerate(zip(LOSSES_NAME, loss_meters)):
                        if task_dict[loss_name] == task:
                            if task == 'vis_mask':
                                if loss_name == 'Mask_Code':
                                    loss_meter.update(results['code_loss'].item())
                                elif loss_name == 'Perceptual':
                                    loss_meter.update(results['perceptual_loss'].item())
                                elif loss_name == 'Recon':
                                    loss_meter.update(results['pixel_loss'].item())

                                if self.args.gan:
                                    if loss_name == 'D_feat_match':
                                        loss_meter.update(results['g_loss_feat_match'].item())
                                    elif loss_name == 'D(fake)':
                                        loss_meter.update(results['D(fake)'].item())
                                    elif loss_name == 'D(real)':
                                        loss_meter.update(results['D(real)'].item())
                                    elif loss_name == 'D loss':
                                        loss_meter.update(results['d_loss'].item())
                                    elif loss_name == 'G loss':
                                        loss_meter.update(results['g_loss_fake'].item())

                            elif task == 'word_mask':
                                loss_meter.update(results['lm_loss'])
                            elif task == 'matched':
                                loss_meter.update(results['matched_loss'])
                            loss_counts[i] += 1
                        if len(loss_meter) > 0:
                            loss_count = loss_counts[i]
                            if loss_name in ['Mask_LM', 'Matched', 'Mask_Code']:
                                desc_str += f' {loss_name} ({loss_count}) {loss_meter.val:.2f}'
                            else:
                                desc_str += f' {loss_name} {loss_meter.val:.2f}'
                    pbar.set_description(desc_str)
                    pbar.update(1)

                dist.barrier()

            if self.verbose:
                pbar.close()

            dist.barrier()

            return epoch_results

    def norm(self, x):
        """(0, 1) => (-1, 1)"""
        out = 2 * x - 1
        return out.clamp(-1, 1)

    def denorm(self, x):
        """(-1, 1) => (0, 1)"""
        out = (x + 1) / 2
        return out.clamp(0, 1)

    def imagenet_norm(self, x):
        imagenet_mean = torch.tensor(
            [0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        imagenet_std = torch.tensor(
            [0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        return (x - imagenet_mean) / imagenet_std

    def imagenet_denorm(self, x):
        imagenet_mean = torch.tensor(
            [0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        imagenet_std = torch.tensor(
            [0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        return x * imagenet_std + imagenet_mean

    def sample_caption(self, visn_feats, seq_length=None, max_seq_length=20, prefix=None, n_steps=None, method='LR'):
        if seq_length is None:
            seq_length = max_seq_length

        B = len(visn_feats[0])
        self.model.eval()

        with torch.no_grad():

            # input_ids = ['[CLS]'] * B
            # input_ids = self.tokenizer.convert_tokens_to_ids(input_ids)
            # input_ids = torch.LongTensor(input_ids).cuda()
            # input_ids = input_ids.view(B, 1)

            # mask_input_ids = ['[MASK]'] * B
            # mask_input_ids = self.tokenizer.convert_tokens_to_ids(mask_input_ids)
            # mask_input_ids = torch.LongTensor(mask_input_ids).cuda()
            # mask_input_ids = mask_input_ids.view(B, 1)

            mask_token_id = self.tokenizer.convert_tokens_to_ids(['[MASK]'])[0]

            # input_ids = ['[CLS]'] + ['[MASK]'] * (seq_length-2) + ['[SEP]']

            if prefix is None:
                input_ids = self.tokenizer.encode('')
                prefix_len = 0
            else:
                input_ids = self.tokenizer.encode(prefix)
                prefix_len = len(input_ids) - 2

            input_ids = input_ids[:-1] + [mask_token_id] * \
                (seq_length - len(input_ids)) + input_ids[-1:]

            input_ids = [input_ids] * B
            input_ids = torch.LongTensor(input_ids).cuda()
            assert input_ids.size() == (B, seq_length), input_ids.size()

            if n_steps is None:
                n_steps = seq_length - 2

            if method != 'LR':
                init_positions = list(range(seq_length))[1:-1]
                random.shuffle(init_positions)

                extra_positions = [random.randint(1 + prefix_len, seq_length-2) for _ in range(n_steps - len(init_positions))]
                positions = extra_positions + init_positions
            for i in range(n_steps):


                if method == 'LR':
                    update_position = i
                else:
                    update_position = positions.pop()

                # if update_position <= prefix_len:
                #     continue

                # Add Mask
                # input_ids = torch.cat([input_ids, mask_input_ids.detach().clone()], dim=1)

                # for random sampling
                input_ids[:, update_position] = mask_token_id

                # language Inputs
                segment_ids = torch.zeros_like(input_ids)
                input_mask = input_ids > 0
                # input_masks = torch.ones_like(input_ids)
                # input_masks[:, i+1:] = 0

                sent_feats = (
                    input_ids,
                    segment_ids,
                    input_mask,
                )

                out_dict = self.model(visn_feats, sent_feats, task='word_mask', calc_loss=False, vis_AR=False)

                lang_logit = out_dict['word_logit']

                # assert lang_logit.size()[:2] == (B, i+2), (i, lang_logit.size())
                assert lang_logit.size()[:2] == (B, seq_length), (i, lang_logit.size())

                prob = torch.softmax(lang_logit[:, update_position], dim=-1)  # [B, vocab_size]
                max_prob, max_idx = prob.max(dim=-1) # [B]

                assert max_idx.size() == (B, ), max_idx.size()

                # Update masked word
                input_ids[:, update_position] = max_idx

        assert input_ids.size() == (B, seq_length), input_ids.size()

        generated_ids = input_ids.tolist()
        generated_sentences = []
        for i in range(B):
            token_ids = generated_ids[i]
            assert len(token_ids) == seq_length, len(token_ids)

            sentence = self.tokenizer.decode(token_ids, skip_special_tokens=True)

            # sentence = self.tokenizer.convert_ids_to_tokens(token_ids)
            # if '[SEP]' in sentence:
            #     last_idx = sentence.index('[SEP]')
            #     sentence = sentence[:last_idx]

            # sentence = sentence[1:]

            # generated_sentences.append(" ".join(sentence))
            generated_sentences.append(sentence)

        return generated_sentences



    def img_sample_parallel(self, sent_feats, n_steps, out_intermediate=False, regress=False):
        grid_size = self.args.grid_size
        code_dim = self.args.codebook_dim

        B = sent_feats[0].size(0)

        grid = torch.from_numpy(box_position(grid_size)).unsqueeze(0).expand(B,-1,-1).cuda()

        intermediate_imgs = []

        for i in range(n_steps):

            # print('iter', i)

            ratio = (n_steps - i) / n_steps
            n_mask = int(ratio * grid_size**2)

            if i == 0:
                vis_mask = torch.ones(B, grid_size**2).long().cuda()
                code = torch.zeros(B, grid_size**2, code_dim).cuda()
            else:
                # [B, n_grids]
                lowest_prob, lowest_arg = pred_prob.topk(n_mask, dim=1, largest=False)
                vis_mask = torch.zeros(B, grid_size**2).long().cuda()
                vis_mask.scatter_(1, lowest_arg, 1)

            visn_feats = (
                code.view(B, grid_size**2, code_dim),
                grid.view(B, grid_size**2, 4),
                vis_mask.view(B, grid_size**2)
            )

            # 2) LXMERT Forward Propagation
            out_dict = self.model(visn_feats, sent_feats, task='vis_mask', calc_loss=False, vis_AR=False)

            if regress:
                pred_code = out_dict['feat']
            else:

                # [B, n_grids, n_codes]
                pred_code_logit = out_dict['code_logit']
                pred_code_prob = torch.softmax(pred_code_logit, dim=2)

                # [B, n_grids]
                pred_prob, pred_prob_arg = pred_code_prob.max(dim=2)

                # [B, n_grids, code_dim]
                pred_code = out_dict['pred_code']

            # 3) Update masked codes
            if self.args.VMP_smart:
                code = pred_code
            else:
                code = torch.where(vis_mask.view(B, grid_size**2, 1).bool(),
                                   pred_code,
                                   code)

            if out_intermediate:
                if type(self.model) in [DDP, DataParallel]:
                    fake_img = self.model.module.G(code.permute(0,2,1).view(B, code_dim, grid_size, grid_size))
                else:
                    fake_img = self.model.G(code.permute(0,2,1).view(B, code_dim, grid_size, grid_size))
                fake_img = self.denorm(fake_img).cpu()
                intermediate_imgs.append(fake_img)

        if out_intermediate:
            return intermediate_imgs

        # 4) Generate image with Pre-trained Decoder
        if type(self.model) in [DDP, DataParallel]:
            fake_img = self.model.module.G(code.permute(0,2,1).view(B, code_dim, grid_size, grid_size))
        else:
            fake_img = self.model.G(code.permute(0,2,1).view(B, code_dim, grid_size, grid_size))
        fake_img = self.denorm(fake_img).cpu()

        return fake_img

    def img_sample_single(self, sent_feats, n_steps=None, out_intermediate=False, seed=None):
        grid_size = self.args.grid_size
        code_dim = self.args.codebook_dim

        B = sent_feats[0].size(0)

        grid = torch.from_numpy(box_position(grid_size)).unsqueeze(
            0).expand(B, -1, -1).cuda()

        intermediate_imgs = []

        if n_steps is None:
            n_steps = grid_size ** 2

        if self.args.sample_random:
            positions = list(range(grid_size ** 2))
            if seed is not None:
                random.Random(seed).shuffle(positions)
            else:
                random.shuffle(positions)
            if n_steps > grid_size ** 2:
                additional_positions = list(range(n_steps - grid_size ** 2))
                if seed is not None:
                    random.Random(seed).shuffle(additional_positions)
                else:
                    random.shuffle(additional_positions)
                positions = additional_positions + positions

        if self.args.sample_confidence:
            visited_positions = torch.zeros(B, grid_size**2).cuda()

        for i in range(n_steps):

            if i == 0:
                vis_mask = torch.ones(B, grid_size**2).long().cuda()
                code = torch.zeros(B, grid_size**2, code_dim).cuda()

            if self.args.sample_random:
                current_pos_i = positions.pop()
                current_pos_i = current_pos_i % grid_size**2
                # for more than NxN iteration
                vis_mask[:, current_pos_i] = 1
            elif self.args.sample_AR:
                current_pos_i = i

            visn_feats = (
                code.view(B, grid_size**2, code_dim),
                grid.view(B, grid_size**2, 4),
                vis_mask.view(B, grid_size**2)
            )

            # 2) LXMERT Forward Propagation
            out_dict = self.model(visn_feats, sent_feats,
                                  task='vis_mask', calc_loss=False, vis_AR=False)

            # [B, n_grids, n_codes]
            pred_code_logit = out_dict['code_logit']
            pred_code_prob = torch.softmax(pred_code_logit, dim=2)

            # [B, n_grids]
            pred_prob, pred_prob_arg = pred_code_prob.max(dim=2)

            # [B, n_grids, code_dim]
            pred_code = out_dict['pred_code']

            # 3) Update masked codes

            # pixelcnn-style
            if self.args.sample_AR or self.args.sample_random:
                update_mask = torch.zeros(B, grid_size**2).bool().cuda()
                update_mask[:, current_pos_i] = 1
                vis_mask[:, current_pos_i] = 0

            elif self.args.sample_confidence:

                _pred_prob = pred_prob.masked_fill(visited_positions.bool(), -10000)

                top_prob, top_arg = _pred_prob.topk(1, dim=1, largest=True)
                update_mask = torch.zeros(B, grid_size**2).long().cuda()
                update_mask.scatter_(1, top_arg, 1)
                vis_mask.scatter_(1, top_arg, 0)

                visited_positions.scatter_(1, top_arg, 1)

            code = torch.where(update_mask.view(B, grid_size**2, 1).bool(),
                               pred_code,
                               code)

            if out_intermediate:
                if type(self.model) in [DDP, DataParallel]:
                    fake_img = self.model.module.G(code.permute(
                        0, 2, 1).view(B, code_dim, grid_size, grid_size))
                else:
                    fake_img = self.model.G(code.permute(0, 2, 1).view(
                        B, code_dim, grid_size, grid_size))
                fake_img = self.denorm(fake_img).cpu()
                intermediate_imgs.append(fake_img)

        if out_intermediate:
            return intermediate_imgs

        # 4) Generate image with Pre-trained Decoder
        if type(self.model) in [DDP, DataParallel]:
            fake_img = self.model.module.G(code.permute(
                0, 2, 1).view(B, code_dim, grid_size, grid_size))
        else:
            fake_img = self.model.G(code.permute(0, 2, 1).view(
                B, code_dim, grid_size, grid_size))
        fake_img = self.denorm(fake_img).cpu()

        return fake_img

    def sample(self, batch_entry, epoch=-1, step_i=-1, n_steps=None, pbar=None, tensorboard=False, verbose=False,
               out_intermediate=None,
              custom_img=False, seed=None, save_sent=True, return_imgs=False, regress=False):

        self.model.eval()
        if save_sent:
            sent_path = self.img_log_dir.joinpath('sentences.txt')
            if not sent_path.exists():
                with open(sent_path, 'w') as f:
                    for i, sent in enumerate(batch_entry['sent']):
                        f.write(f'{i}. {sent}\n')



        if out_intermediate is None:
            out_intermediate = self.args.MP_out_intermediate

        with torch.no_grad():
            visn_feats, sent_feats, _ = self.encode(batch_entry, 'vis_mask')
            B = visn_feats[0].size(0)

            if n_steps is None:
                if self.args.sample_single_grid:
                    n_steps = self.args.grid_size ** 2
                else:
                    n_steps = 4

            if out_intermediate:
                if self.args.sample_single_grid:
                    fake_MP_img_list = self.img_sample_single(sent_feats, n_steps=n_steps, out_intermediate=True, seed=seed)
                else:
                    fake_MP_img_list = self.img_sample_parallel(
                        sent_feats, n_steps=n_steps, out_intermediate=True, regress=regress)

                if return_imgs:
                    return fake_MP_img_list

                for i, fake_MP_img in enumerate(fake_MP_img_list):
                    assert fake_MP_img.size() == (B, 3, self.args.resize_target_size, self.args.resize_target_size), fake_MP_img.size()

                    fake_MP_img_path = self.img_log_dir.joinpath(
                        f"fake_MP_epoch{epoch:02d}_step{step_i:04d}_iter{i}_total{n_steps}.png")

                    if verbose:
                        if pbar is not None:
                            pbar.write(f'Save fake MP images at {fake_MP_img_path}')
                    save_image(fake_MP_img, fake_MP_img_path)
                    if tensorboard:
                        global_step_i = epoch * len(self.train_loader) + step_i
                        self.writer.add_image(f'fake_MP_iter{i}', make_grid(fake_MP_img), global_step_i)
            else:
                if self.args.sample_single_grid:
                    fake_MP_img = self.img_sample_single(sent_feats, n_steps=n_steps, out_intermediate=False, seed=seed)
                else:
                    fake_MP_img = self.img_sample_parallel(
                        sent_feats, n_steps=n_steps, out_intermediate=False, regress=regress)
                assert fake_MP_img.size() == (B, 3, self.args.resize_target_size, self.args.resize_target_size), fake_MP_img.size()

                if return_imgs:
                    return fake_MP_img

                fake_MP_img_path = self.img_log_dir.joinpath(
                    f"fake_MP_epoch{epoch:02d}_step{step_i:04d}.png")

                if verbose:
                    if pbar is not None:
                        pbar.write(f'Save fake MP images at {fake_MP_img_path}')
                save_image(fake_MP_img, fake_MP_img_path)
                if tensorboard:
                    global_step_i = epoch * len(self.train_loader) + step_i
                    self.writer.add_image('fake_MP', make_grid(fake_MP_img), global_step_i)

            if not custom_img:

                # Get reconstruction from gt codes
                gt_code = visn_feats[0].permute(0,2,1).view(B, self.args.codebook_dim, self.args.grid_size, self.args.grid_size)
                if type(self.model) in [DDP, DataParallel]:
                    fake_recon_img = self.model.module.G(gt_code)
                else:
                    fake_recon_img = self.model.G(gt_code)

                fake_recon_img = self.denorm(fake_recon_img)  # [-1, 1] => [0, 1]
                assert fake_recon_img.size() == (B, 3, self.args.resize_target_size, self.args.resize_target_size), fake_recon_img.size()

                fake_recon_img_path = self.img_log_dir.joinpath(
                    f"fake_recon_epoch{epoch:02d}_step{step_i:04d}.png")

                if verbose:
                    if pbar is not None:
                        pbar.write(f'Save fake recon_images at {fake_recon_img_path}')
                save_image(fake_recon_img, fake_recon_img_path)
                if tensorboard:
                    global_step_i = epoch * len(self.train_loader) + step_i
                    self.writer.add_image('fake_recon', make_grid(fake_recon_img), global_step_i)

                real_img_path = self.img_log_dir.joinpath("real.png")
                if not real_img_path.exists():
                    if verbose:
                        if pbar is not None:
                            pbar.write(f'Save real images at {real_img_path}')
                    real_img = batch_entry['img']
                    save_image(real_img, real_img_path)
                    if tensorboard:
                        self.writer.add_image('real', make_grid(real_img), 0)

    def save(self, name):
        torch.save(self.model.state_dict(),
                   os.path.join(self.output, "%s.pth" % name))
        torch.save(self.D.state_dict(),
                   os.path.join(self.output, "D_%s.pth" % name))

    def load(self, path, loc=None, verbose=True, strict=False):
        print("Load model from %s" % path)
        if loc is None:
            state_dict = torch.load("%s.pth" % path)
        else:
            state_dict = torch.load("%s.pth" % path, map_location=loc)

        # Change Multi GPU to single GPU
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                new_state_dict[key[len("module."):]] = value
        state_dict = new_state_dict
        if verbose:
            load_keys = set(state_dict.keys())
            model_keys = set(self.model.state_dict().keys())
            print()
            print("Keys in loaded but not in model:")
            for key in sorted(load_keys.difference(model_keys)):
                print(key)
            print()
            print("Keys in model but not in loaded:")
            for key in sorted(model_keys.difference(load_keys)):
                print(key)
            print()

        results = self.model.load_state_dict(state_dict, strict=strict)
        if verbose:
            print(results)

        # self.model.load_state_dict(state_dict)


class LossMeter(object):
    def __init__(self, maxlen=100):
        """Computes and stores the running average"""
        self.vals = collections.deque([], maxlen=maxlen)

    def __len__(self):
        return len(self.vals)

    def update(self, new_val):
        self.vals.append(new_val)

    @property
    def sum(self):
        return sum(self.vals)

    @property
    def val(self):
        return sum(self.vals) / len(self.vals)

    def __repr__(self):
        return str(self.val)


def main_worker(gpu, args):
    # GPU is assigned
    args.gpu = gpu
    args.rank = gpu
    print(f'Process Launching at GPU {gpu}')

    if args.distributed:
        torch.cuda.set_device(args.gpu)
        dist.init_process_group(backend='nccl',
                                init_method='tcp://127.0.0.1:9000',
                                # init_method="env://",
                                world_size=args.world_size,
                                rank=args.rank)

    # Logging
    if gpu == 0:
        logger = logging.getLogger('mylogger')
        file_handler = logging.FileHandler(args.log_dir.joinpath('log.txt'))
        # stream_handler = logging.StreamHandler()
        logger.addHandler(file_handler)
        # logger.addHandler(stream_handler)
        logger.setLevel(logging.DEBUG)
        args.save(args.log_dir.joinpath('args.yaml'))
        logger.info(str(args))

        # Save scripts for backup
        log_src_dir = args.log_dir.joinpath(f'src/')
        src_dir = Path(__file__).resolve().parent.parent
        if log_src_dir.exists():
            shutil.rmtree(log_src_dir)
        shutil.copytree(src_dir, log_src_dir)
        print('Source files logged at', log_src_dir)
        logger.info('Source files logged at' + str(log_src_dir))

    else:
        logger = None

    train_transform = transforms.Compose([
        # transforms.RandomResizedCrop(
        #     (args.imsize, args.imsize),
        #     scale=(0.3, 1.0),
        #     interpolation=Image.LANCZOS
        # ),
        transforms.Resize(
            (args.resize_input_size, args.resize_input_size),
            interpolation=Image.LANCZOS
        ),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(
        #     [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        # )
    ])
    valid_transform = transforms.Compose([
        transforms.Resize(
            (args.resize_input_size, args.resize_input_size),
            interpolation=Image.LANCZOS
        ),
        transforms.ToTensor(),
        # transforms.Normalize(
        #     [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        # )
    ])

    data_out = ['img', 'sent']
    if args.task_matched:
        data_out += ['matched']
    if args.clustering:
        data_out += ['cluster_id']
    if args.word_mask_predict:
        data_out += ['word_mask_idx']
    if args.vis_mask_predict:
        if not args.square_mask:
            data_out += ['vis_mask_idx']

    if gpu == 0:
        print('Dataloader output:', data_out)

    if not args.debug:
        train_loader = get_loader(
            args,
            split=args.train, mode='train', batch_size=args.batch_size,
            distributed=args.distributed, gpu=args.gpu,
            workers=args.num_workers, transform=train_transform,
            topk=args.train_topk,
            data_out=data_out
        )

        val_loader = get_loader(
            args,
            split=args.valid, mode='val', batch_size=args.batch_size,
            distributed=args.distributed,
            gpu=args.gpu,
            workers=args.num_workers, transform=valid_transform,
            topk=args.valid_topk,
            data_out=data_out
        )

        trainer = ImgGen(args, train_loader, val_loader, logger, train=True)
        trainer.train()

    else:
        train_loader = []
        val_loader = get_loader(
            args,
            split=args.valid, mode='val', batch_size=args.batch_size,
            # distributed=args.distributed,
            distributed=True,
            gpu=args.gpu,
            workers=args.num_workers, transform=valid_transform,
            topk=args.valid_topk,
            data_out=data_out
        )

        trainer = ImgGen(args, train_loader, val_loader, None, False)

        trainer.img_log_dir = args.log_dir.joinpath('images')
        trainer.img_log_dir.mkdir(exist_ok=True)

        batch_entry = next(iter(val_loader))
        print(batch_entry['sent'])
        trainer.sample(batch_entry, -1, -1, tensorboard=False)


if __name__ == "__main__":
    # cudnn.benchmark = True
    args = parse_args()
    print(args)
    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node
    args.n_grids = args.grid_size ** 2

    LOSSES_NAME = []

    if args.task_mask_lm:
        LOSSES_NAME.append('Mask_LM')
    if args.task_matched:
        LOSSES_NAME.append('Matched')
    if 'img' in args.visual_losses.split(','):
        # LOSSES_NAME.append('Mask_Vis')
        LOSSES_NAME.append('Mask_Code')
        LOSSES_NAME.append('Perceptual')
        LOSSES_NAME.append('Recon')

        if args.gan:
            LOSSES_NAME.extend([
                'D_feat_match', 'D(fake)', 'D(real)', 'D loss', 'G loss'])

    MASK_MODALITY = []
    if args.task_obj_predict:
        MASK_MODALITY.append('vis_mask')
    if args.task_mask_lm:
        MASK_MODALITY.append('word_mask')
    if args.task_matched:
        MASK_MODALITY.append('matched')

    print('Losses:', LOSSES_NAME)
    print('Mask modalities:', MASK_MODALITY)
    args.LOSSES_NAME = LOSSES_NAME
    args.MASK_MODALITY = MASK_MODALITY

    from datetime import datetime

    comment = f'Grid{args.grid_size}'
    if args.clustering:
        comment += '_cluster'
    if args.backbone == 'uniter':
        comment += '_uniter'
    if 'mscoco_train' in args.train:
        comment += '_COCO'
        if 'vgnococo' in args.train:
            comment += 'VG'
    if args.from_scratch:
        comment += '_fromscratch'
    if args.load_lxmert is not None:
        comment += f'_{args.load_lxmert}'
    if args.word_mask_predict:
        comment += f'_WMP'
    elif args.word_mask_rate > 0:
        comment += f'_W{int(args.word_mask_rate*100)}'
    if args.vis_mask_predict:
        comment += f'_VMP'
    elif args.obj_mask_rate > 0:
        comment += f'_V{int(args.obj_mask_rate*100)}'
    if args.task_matched:
        comment += f'_Matched'
    # if args.clustering:
    #     comment += f'_Vocab{args.n_centroids}'
    comment += f'_imsize{args.resize_target_size}'

    if args.comment != '':
        comment += f'_{args.comment}'

    current_time = datetime.now().strftime('%b%d_%H-%M')
    project_dir = Path(__file__).resolve().parent.parent.parent
    log_dir = project_dir.joinpath('runs')
    log_dir = log_dir.joinpath(
        'ImgGen_Joint', current_time + f'_GPU{args.world_size}_' + comment)
    args.log_dir = log_dir
    log_dir.mkdir(exist_ok=True, parents=True)
    print('logging at', log_dir)

    if args.distributed:
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(args,))
    else:
        main_worker(0, args)
