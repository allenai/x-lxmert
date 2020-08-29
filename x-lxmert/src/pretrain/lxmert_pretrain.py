# coding=utf-8
# Copyleft 2019 project LXRT.

import collections
import os
import random
from pathlib import Path
import logging
import shutil

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn

from param import args
# from param import parse_args

from lxrt.entry import set_visual_config
from lxrt.modeling import LXRTPretraining
from lxrt.tokenization import BertTokenizer
# try:
#     from lxmert_data import get_loader
# except:
from pretrain.lxmert_data import get_loader#, cluster_src

# from resnet_feat_config import feat_dim, n_class, n_centroids, n_iter

from apex import amp
from torch.nn.parallel import DistributedDataParallel as DDP
# from torchvision import transforms
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def reduce_dict(_results, gpu=0, reduce_gpu=0, long=False):
    results = {}
    for k, v in _results.items():
        if type(v) == torch.Tensor:
            if long:
                results[k] = v.long().cuda(gpu)
            else:
                results[k] = v.float().cuda(gpu)
        else:
            if long:
                results[k] = torch.tensor(v).long().cuda(gpu)
            else:
                results[k] = torch.tensor(v).float().cuda(gpu)

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


class LXMERT:
    def __init__(self, args, train_loader=None, val_loader=None, logger=None, train=True):
        super().__init__()

        self.args = args
        self.max_text_length = args.max_text_length

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.logger = logger

        # Build model
        set_visual_config(args)
        self.model = LXRTPretraining.from_pretrained(
            "bert-base-uncased",
            task_mask_lm=args.task_mask_lm,
            task_obj_predict=args.task_obj_predict,
            task_matched=args.task_matched,
            task_qa=args.task_qa,
            visual_losses=args.visual_losses,
            num_answers=self.train_loader.dataset.answer_table.num_answers if self.args.task_qa else 2,
            gpu=args.gpu,
            args=args
        )

        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased",
            do_lower_case=True
        )

        self.verbose = True
        if self.args.distributed:
            if self.args.gpu != 0:
                self.verbose = False

        if args.clustering:
            if args.vis_vocab_from_scratch:
                Emb = nn.Embedding(args.n_centroids, args.feat_dim)
            else:
                centroid_dir = Path(
                    '/home/jaeminc/Dropbox/Projects/AI2/clustering/').resolve()
                if args.v4:
                    centroid_dir = centroid_dir.joinpath('v4')
                imsize = args.resize_input_size
                if args.im_ratio == 'original':
                    centroid_path = centroid_dir.joinpath(
                        f'{args.encoder}_{args.cluster_src}_centroids{args.n_centroids}_iter{args.n_iter}_d{args.feat_dim}_grid{args.grid_size}.npy')
                else:
                    centroid_path = centroid_dir.joinpath(
                        f'{args.encoder}_{args.cluster_src}_centroids{args.n_centroids}_iter{args.n_iter}_d{args.feat_dim}_grid{args.grid_size}_imsize{imsize}.npy')
                centroids = np.load(centroid_path)

                Emb = nn.Embedding.from_pretrained(
                    torch.from_numpy(centroids),
                    freeze=True
                    # freeze=False
                )

            self.vis_emb = Emb
            # self.model.mask_feat = nn.Parameter(
            #     torch.zeros(self.vis_emb.weight.size(1)))
            if self.args.task_obj_predict:
                # self.model.obj_predict_head.decoder_dict['obj'][1].weight = self.vis_emb.weight
                self.model.obj_predict_head.out_cluster.weight = self.vis_emb.weight
        else:
            Emb = None
            self.vis_emb = Emb

            if self.args.task_obj_predict and self.args.target_cluster:
                centroid_dir = Path(
                    '/home/jaeminc/Dropbox/Projects/AI2/clustering/').resolve()
                if args.v4:
                    centroid_dir = centroid_dir.joinpath('v4')
                imsize = args.resize_input_size
                if args.im_ratio == 'original':
                    centroid_path = centroid_dir.joinpath(
                        f'{args.encoder}_{args.cluster_src}_centroids{args.n_centroids}_iter{args.n_iter}_d{args.feat_dim}_grid{args.grid_size}.npy')
                else:
                    centroid_path = centroid_dir.joinpath(
                        f'{args.encoder}_{args.cluster_src}_centroids{args.n_centroids}_iter{args.n_iter}_d{args.feat_dim}_grid{args.grid_size}_imsize{imsize}.npy')
                centroids = np.load(centroid_path)

                centroids = torch.from_numpy(centroids)
                centroids.requires_grad_(False)

                self.model.obj_predict_head.out_cluster.weight.data = centroids

        # for name, p in self.model.named_parameters():
        #     print(name, '\t', list(p.size()), 'requires_grad', p.requires_grad)
        # print(count_parameters(self.model))
        # exit()


        # Weight initialization
        if args.from_scratch:
            if args.gpu == 0:
                print("Train from Scratch: re-initialize all BERT weights.")
            self.model.apply(self.model.init_bert_weights)

        # Load pre-trained weights
        self.start_epoch = None
        if args.load is not None:
            lxmert_ckpt = args.load +'_LXRT.pth'
            state_dict = load_state_dict(lxmert_ckpt, 'cpu')
            results = self.model.load_state_dict(state_dict, strict=False)
            if self.verbose:
                print('LXRT loaded from ', lxmert_ckpt)
                print(results)
            self.start_epoch = int(args.load.split('Epoch')[-1])

        # GPU Options
        print(f'Model Launching at GPU {self.args.gpu}')

        from time import time
        start = time()
        self.model = self.model.cuda(args.gpu)
        if args.clustering:
            self.vis_emb = self.vis_emb.cuda(args.gpu)

        # Optimizer
        self.scheduler = None
        if train:
            if args.gpu == 0:
                print('Building Optimizer')
            if 'bert' in args.optim:
                from lxrt.optimization import BertAdam
                batch_per_epoch = len(self.train_loader)
                t_total = int(batch_per_epoch * self.args.epochs)
                warmup_ratio = 0.05
                warmup_iters = int(t_total * warmup_ratio)
                if args.gpu == 0:
                    print("Batch per epoch: %d" % batch_per_epoch)
                    print("Total Iters: %d" % t_total)
                    print('Warmup ratio:', warmup_ratio)
                    print("Warm up Iters: %d" % warmup_iters)

                # params = [p for p in self.model.parameters() if p.requires_grad]
                params = self.model.parameters()
                self.optim = BertAdam(params, lr=self.args.lr,
                                    warmup=warmup_ratio, t_total=t_total)
            elif 'adamw' in args.optim:
                from transformers.optimization import AdamW, get_linear_schedule_with_warmup
                batch_per_epoch = len(train_loader)
                t_total = int(batch_per_epoch * args.epochs)
                warmup_ratio = 0.05
                warmup_iters = int(t_total * warmup_ratio)
                if self.verbose:
                    print("Batch per epoch: %d" % batch_per_epoch)
                    print("Total Iters: %d" % t_total)
                    print('Warmup ratio:', warmup_ratio)
                    print("Warm up Iters: %d" % warmup_iters)

                no_decay = ["bias", "LayerNorm.weight"]
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                        "weight_decay": True,
                    },
                    {
                        "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                        "weight_decay": 0.0,
                    },
                ]

                self.optim = AdamW(optimizer_grouped_parameters, args.lr)

            if args.mixed_precision:
                if args.clustering:
                    [self.model, self.vis_emb], self.optim = amp.initialize(
                        [self.model, self.vis_emb], self.optim, opt_level='O1', verbosity=self.verbose)
                else:
                    self.model, self.optim = amp.initialize(
                        self.model, self.optim, opt_level='O1', verbosity=self.verbose)
        else:
            self.optim = None

        if 'adamw' in args.optim:
            self.scheduler = get_linear_schedule_with_warmup(self.optim, warmup_iters, t_total)


        if args.multiGPU:
            # self.model = nn.DataParallel(self.model)
            if args.distributed:
                self.model = DDP(self.model, device_ids=[args.gpu],
                                 find_unused_parameters=True
                                 )
        if args.gpu == 0:
            print(f'It took {time() - start:.1f}s')


        # # Load Checkpoint
        # self.start_epoch = None
        # if args.load is not None:
        #     loc = 'cuda:{}'.format(args.gpu)
        #     self.load(args.load, loc=loc)
        # if args.load_lxmert is not None:
        #     # Load lxmert would not load the answer head.
        #     loc = 'cuda:{}'.format(args.gpu)
        #     self.load_lxmert(args.load_lxmert, loc=loc)
        #     self.start_epoch = int(args.load_lxmert.split('Epoch')[-1])

        cudnn.benchmark = True

    def forward(self, batch, task):
        if self.args.feed_exact_feat:
            vis_feats = batch['vis_feats'].clone().detach().cuda()
        if self.args.clustering or self.args.target_cluster:
            cluster_id = batch['cluster_id'].cuda()
        if self.args.clustering:
            vis_feats = self.vis_emb(cluster_id)

        vis_mask = batch['vis_mask'].cuda().bool()

        label_dict = {}
        if task == 'word_mask':
            label_dict['word_labels'] = batch['word_label'].cuda()
        elif task == 'vis_mask':
            if 'obj' in self.args.visual_losses:
                if self.args.clustering or self.args.target_cluster:
                    obj_labels = cluster_id.clone().detach()
                    obj_labels[~vis_mask] = -1
                    label_dict['obj_labels'] = obj_labels
                elif self.args.target_obj_id:
                    obj_labels = batch['obj_ids'].cuda()
                    obj_labels[~vis_mask] = -1
                    label_dict['obj_labels'] = obj_labels

                    # # obj_probs = batch['obj_probs'].cuda()
                    # label_dict['obj_labels'] = obj_probs
                    # obj_labels = batch['obj_label'].cuda()
                    # obj_labels[~vis_mask] = -1
                    # label_dict['obj_labels'] = obj_labels
            # if 'attr' in self.args.visual_losses:
            #     attr_labels = batch['attr_label'].cuda()
            #     attr_labels[~vis_mask] = -1
            #     label_dict['attr_labels'] = attr_labels

                # attr_probs = batch['attr_probs'].cuda()
                # label_dict['attr_labels'] = attr_probs
            if 'feat' in self.args.visual_losses:
                vis_feats = batch['vis_feats'].clone().detach().cuda()
                label_dict['feat_labels'] = vis_feats
        elif task == 'matched':
            matched_label = batch['matched_label'].cuda()
            label_dict['matched_labels'] = matched_label
        # elif task == 'qa':
        if self.args.task_qa:
            qa_label = batch['qa_label'].cuda()
            if task == 'matched':
                flipped = (matched_label == 0)
                qa_label.masked_fill_(flipped, -1)
            label_dict['qa_labels'] = qa_label

        # print(f'GPU {self.args.gpu} label_dict', label_dict)

        visn_feats = (
            vis_feats,
            batch['box_position'].cuda(),
            vis_mask
        )

        # print('input visn_feats', visn_feats)

        if task == 'word_mask':
            word_id = batch['masked_word_id'].cuda()
        elif task == 'matched':
            word_id = batch['other_word_id'].cuda()
        elif task == 'vis_mask':
            word_id = batch['word_id'].cuda()
        # elif task == 'qa':
        #     word_id = batch['word_id'].cuda()

        word_attention_mask = word_id > 0
        token_type_ids = torch.zeros_like(word_id)
        sent_feats = (
            word_id,
            token_type_ids,
            word_attention_mask,
        )

        out_dict = self.model(
            visn_feats,
            sent_feats,
            label_dict=label_dict,
            task=task,
            calc_loss=True,
            vis_AR=self.args.visual_AR)

        return out_dict

    def train(self):
        LOSSES_NAME = self.args.LOSSES_NAME
        task_dict = {
            'Mask_LM': 'word_mask',
            'Matched': 'matched',
            'Mask_Obj': 'vis_mask',
            'Mask_Attr': 'vis_mask',
            'Mask_Feat': 'vis_mask',
            'QA': 'qa'
        }

        if self.args.dry:
            results = self.evaluate_epoch(epoch=0)

        self.optim.zero_grad()

        if self.verbose:
            loss_meters = [LossMeter() for _ in range(len(LOSSES_NAME))]
            best_eval_loss = 9595.

            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=self.args.log_dir)
            print('logging at', str(self.args.log_dir))
            self.logger.info('logging at' + str(self.args.log_dir))

            hparam_dict = {}
            for k, v in self.args.__dict__.items():
                if type(v) in [int, float, str, bool, torch.Tensor]:
                    hparam_dict[k] = v
            metric_dict = {}

            self.writer.add_hparams(hparam_dict, metric_dict)

        dist.barrier()

        n_update = 0
        global_step = 0
        for epoch in range(self.args.epochs):
            if self.start_epoch is not None:
                epoch += self.start_epoch
            if self.args.distributed:
                self.train_loader.sampler.set_epoch(epoch)

            # Train
            self.model.train()
            loss_counts = [0 for _ in range(len(LOSSES_NAME))]

            if self.verbose:
                pbar = tqdm(total=len(self.train_loader), ncols=240)

            epoch_results = {
                'lm_loss': 0,
                'vis_loss': 0,
                'matched_loss': 0,

                'qa_loss': 0,

                'obj_loss': 0,
                'feat_loss': 0,
                'attr_loss': 0,
            }
            for k in list(epoch_results.keys()):
                if k[-4:] == 'loss':
                    epoch_results[f'{k}_count'] = 0

            if self.args.task_qa:
                uid2ans = {}

            for step_i, batch in enumerate(self.train_loader):
                # task = random.choice(self.args.MASK_MODALITY)
                task_i = step_i % len(self.args.MASK_MODALITY)
                task = self.args.MASK_MODALITY[task_i]

                if self.args.vis_mask_COCO_only or self.args.vis_mask_COCOVG_only:
                    if task == 'vis_mask':
                        batch['word_id'] = batch['COCO_word_id']
                    if self.args.clustering:
                        batch['cluster_id'] = batch['COCO_cluster_id']

                # with torch.autograd.set_detect_anomaly(True):
                results = self.forward(batch, task)

                if task == 'vis_mask':
                    if 'Mask_Obj' in LOSSES_NAME:
                        epoch_results['obj_loss_count'] += 1
                    if 'Mask_Feat' in LOSSES_NAME:
                        epoch_results['feat_loss_count'] += 1
                    if 'Mask_Attr' in LOSSES_NAME:
                        epoch_results['attr_loss_count'] += 1

                    epoch_results['vis_loss_count'] += 1
                    loss = results['vis_loss']
                elif task == 'word_mask':
                    epoch_results['lm_loss_count'] += 1
                    loss = results['lm_loss']
                elif task == 'matched':
                    epoch_results['matched_loss_count'] += 1
                    loss = results['matched_loss']
                # elif task == 'qa':
                if self.args.task_qa:
                    epoch_results['qa_loss_count'] += 1
                    qa_loss = results['qa_loss']
                    loss = loss + qa_loss

                    qa_pred = results['qa_pred']
                    for uid, ans_id in zip(batch['uid'], qa_pred.cpu().numpy()):
                        ans = self.train_loader.dataset.answer_table.id2ans(ans_id)
                        uid2ans[uid] = ans

                #===== Update =====#
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

                    if not self.args.no_clip_grad:
                        if update:
                            nn.utils.clip_grad_norm_(amp.master_params(self.optim), self.args.clip_grad_norm)
                else:
                    loss.backward()
                    if not self.args.no_clip_grad:
                        if update:
                            nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)

                if update:
                    self.optim.step()
                    if self.scheduler is not None:
                        self.scheduler.step()
                    self.optim.zero_grad()
                    global_step += 1
                #====================#

                for k, v in results.items():
                    if k in epoch_results:
                        epoch_results[k] += v.item()

                if 'bert' in self.args.optim:
                    try:
                        lr = self.optim.lr
                    except AttributeError:
                        lr = 0.
                elif 'adamw' in self.args.optim:
                    try:
                        lr = self.scheduler.get_last_lr()[0]
                    except AttributeError:
                        lr = self.args.lr
                else:
                    lr = self.args.lr

                if self.verbose:
                    desc_str = f'Epoch {epoch} | LR {lr:.6f} | '
                    if self.args.word_mask_predict:
                        desc_str += f'Word Mask: Uniform (MP) | '
                    elif self.args.word_mask_rate > 0:
                        desc_str += f'Word Mask: {self.args.word_mask_rate:.2f} | '

                    if self.args.vis_mask_predict:
                        desc_str += f'Vis Mask: Uniform (MP) |'
                    else:
                        desc_str += f'Vis Mask: {self.args.obj_mask_rate:.2f} |'

                    if self.args.task_qa:
                        loss_meter = loss_meters[-1]
                        loss_meter.update(results['qa_loss'].item())
                        loss_counts[-1] += 1

                    for i, (loss_name, loss_meter) in enumerate(zip(LOSSES_NAME, loss_meters)):
                        if task_dict[loss_name] == task:
                            if task == 'vis_mask':
                                if loss_name == 'Mask_Obj':
                                    loss_meter.update(results['obj_loss'].item())
                                elif loss_name == 'Mask_Attr':
                                    loss_meter.update(results['attr_loss'].item())
                                elif loss_name == 'Mask_Feat':
                                    loss_meter.update(results['feat_loss'].item())
                            elif task == 'word_mask':
                                loss_meter.update(results['lm_loss'].item())
                            elif task == 'matched':
                                loss_meter.update(results['matched_loss'].item())
                            # elif task == 'qa':
                            #     loss_meter.update(results['qa_loss'].item())

                            loss_counts[i] += 1
                        if len(loss_meter) > 0:
                            loss_count = loss_counts[i]
                            if loss_name in ['Mask_LM', 'Matched', 'Mask_Obj', 'Mask_Attr', 'Mask_Feat', 'QA']:
                                desc_str += f' {loss_name} ({loss_count}) {loss_meter.val:.3f}'
                            else:
                                desc_str += f' {loss_name} {loss_meter.val:.3f}'

                            if step_i % 10 == 0:
                                self.writer.add_scalar(f'Train_steps/{loss_name}', loss_meter.val, global_step)

                    if update:
                        n_update += 1
                    desc_str += f' | Total Update: {n_update}'

                    pbar.set_description(desc_str)
                    pbar.update(1)

            if self.verbose:
                pbar.close()

            dist.barrier()

            results = reduce_dict(epoch_results, self.args.gpu)
            if self.args.gpu == 0:
                total_loss = results['lm_loss'] + results['vis_loss'] + results['matched_loss'] + results['qa_loss']
                total_count = results['lm_loss_count'] + results['vis_loss_count'] + results['matched_loss_count'] 
                # + results['qa_loss_count']

                avg_train_loss = total_loss / total_count
                losses_str = f"Train Loss: {avg_train_loss:.4f}\n"

                for name, loss in results.items():
                    if name[-4:] == 'loss':
                        loss_count = int(results[name+'_count'])
                        if loss_count > 0:
                            avg_loss = loss/loss_count
                            if name == 'lm_loss':
                                name = 'Mask_LM'
                            elif name == 'matched_loss':
                                name = 'Matched'
                            elif name == 'obj_loss':
                                name = 'Mask_Obj'
                            elif name == 'attr_loss':
                                name = 'Mask_Attr'
                            elif name == 'feat_loss':
                                name = 'Mask_Feat'
                            elif name == 'qa_loss':
                                name = 'QA'
                            losses_str += f"{name} ({loss_count}): {avg_loss:.4f} "
                            self.writer.add_scalar(f'Train Loss/{name}', avg_loss, epoch)
                losses_str += '\n'
                print(losses_str)
                self.logger.info(losses_str)

            if self.args.task_qa:
                dset2score, dset2cnt, score, cnt = self.train_loader.dataset.evaluator.evaluate(uid2ans)

                dset2score = reduce_dict(dset2score, self.args.gpu)
                dset2cnt = reduce_dict(dset2cnt, self.args.gpu)
                score_cnt_dict = reduce_dict({'score': score, 'cnt': cnt}, self.args.gpu)

                if self.args.gpu == 0:
                    score = score_cnt_dict['score']
                    cnt = score_cnt_dict['cnt']
                    accu = score / cnt
                    dset2accu = {}
                    for dset in dset2cnt:
                        dset2accu[dset] = dset2score[dset] / dset2cnt[dset]
                    accu_str = "Overall Accu %0.4f, " % (accu)
                    sorted_keys = sorted(dset2accu.keys())
                    for key in sorted_keys:
                        accu_str += "%s Accu %0.4f, " % (key, dset2accu[key])
                    print(accu_str)
                    self.logger.info(accu_str)

            dist.barrier()

            # Validation
            valid_results, valid_uid2ans = self.evaluate_epoch(epoch=epoch)

            valid_results = reduce_dict(valid_results, self.args.gpu)
            if self.args.gpu == 0:
                valid_total_loss = valid_results['lm_loss'] + valid_results['vis_loss'] + valid_results['matched_loss'] + valid_results['qa_loss']
                valid_total_count = valid_results['lm_loss_count'] + valid_results['vis_loss_count'] + valid_results['matched_loss_count']
                #  + valid_results['qa_loss_count']

                avg_valid_loss = valid_total_loss / valid_total_count
                losses_str = f"Valid Loss: {avg_valid_loss:.4f}\n"

                for name, loss in valid_results.items():
                    if name[-4:] == 'loss':
                        loss_count = int(valid_results[name+'_count'])
                        if loss_count > 0:
                            avg_loss = loss/loss_count
                            if name == 'lm_loss':
                                name = 'Mask_LM'
                            elif name == 'matched_loss':
                                name = 'Matched'
                            elif name == 'obj_loss':
                                name = 'Mask_Obj'
                            elif name == 'attr_loss':
                                name = 'Mask_Attr'
                            elif name == 'feat_loss':
                                name = 'Mask_Feat'
                            elif name == 'qa_loss':
                                name = 'QA'
                            losses_str += f"{name} ({loss_count}): {avg_loss:.4f} "
                            self.writer.add_scalar(f'Valid Loss/{name}', avg_loss, epoch)

                losses_str += '\n'
                print(losses_str)
                self.logger.info(losses_str)

            if self.args.task_qa:
                dset2score, dset2cnt, score, cnt = self.val_loader.dataset.evaluator.evaluate(valid_uid2ans)

                dset2score = reduce_dict(dset2score, self.args.gpu)
                dset2cnt = reduce_dict(dset2cnt, self.args.gpu)
                score_cnt_dict = reduce_dict({'score': score, 'cnt': cnt}, self.args.gpu)

                if self.args.gpu == 0:
                    score = score_cnt_dict['score']
                    cnt = score_cnt_dict['cnt']
                    accu = score / cnt
                    dset2accu = {}
                    for dset in dset2cnt:
                        dset2accu[dset] = dset2score[dset] / dset2cnt[dset]
                    accu_str = "Overall Accu %0.4f, " % (accu)
                    sorted_keys = sorted(dset2accu.keys())
                    for key in sorted_keys:
                        accu_str += "%s Accu %0.4f, " % (key, dset2accu[key])
                    print(accu_str)
                    self.logger.info(accu_str)

            dist.barrier()

            if self.verbose:
                # Save
                if avg_valid_loss < best_eval_loss:
                    best_eval_loss = avg_valid_loss
                #     self.save("BEST_EVAL_LOSS")
                self.save("Epoch%02d" % (epoch + 1))

            dist.barrier()

    def evaluate_epoch(self, epoch):
        LOSSES_NAME = self.args.LOSSES_NAME
        task_dict = {
            'Mask_LM': 'word_mask',
            'Matched': 'matched',
            'Mask_Obj': 'vis_mask',
            'Mask_Attr': 'vis_mask',
            'Mask_Feat': 'vis_mask',
            'QA': 'qa'
        }

        epoch_results = {
            'lm_loss': 0,
            'vis_loss': 0,
            'matched_loss': 0,

            'qa_loss': 0,

            'obj_loss': 0,
            'feat_loss': 0,
            'attr_loss': 0,
        }
        for k in list(epoch_results.keys()):
            if k[-4:] == 'loss':
                epoch_results[f'{k}_count'] = 0

        uid2ans = {}

        self.model.eval()
        with torch.no_grad():
            if self.verbose:
                loss_meter = LossMeter()
                loss_meters = [LossMeter() for _ in range(len(LOSSES_NAME))]

                loss_counts = [0 for _ in range(len(LOSSES_NAME))]

                pbar = tqdm(total=len(self.val_loader), ncols=180)

            for step_i, batch in enumerate(self.val_loader):
                # task = random.choice(self.args.MASK_MODALITY)
                task_i = step_i % len(self.args.MASK_MODALITY)
                task = self.args.MASK_MODALITY[task_i]
                if self.args.vis_mask_COCO_only or self.args.vis_mask_COCOVG_only:
                    if task == 'vis_mask':
                        batch['word_id'] = batch['COCO_word_id']
                    if self.args.clustering:
                        batch['cluster_id'] = batch['COCO_cluster_id']

                results = self.forward(batch, task)

                if task == 'vis_mask':
                    epoch_results['vis_loss_count'] += 1
                    if 'Mask_Obj' in LOSSES_NAME:
                        epoch_results['obj_loss_count'] += 1
                    if 'Mask_Feat' in LOSSES_NAME:
                        epoch_results['feat_loss_count'] += 1
                    if 'Mask_Attr' in LOSSES_NAME:
                        epoch_results['attr_loss_count'] += 1
                elif task == 'word_mask':
                    epoch_results['lm_loss_count'] += 1
                elif task == 'matched':
                    epoch_results['matched_loss_count'] += 1
                elif task == 'qa':
                    epoch_results['qa_loss_count'] += 1

                if self.args.task_qa:
                    qa_pred = results['qa_pred']
                    for uid, ans_id in zip(batch['uid'], qa_pred.cpu().numpy()):
                        ans = self.train_loader.dataset.answer_table.id2ans(
                            ans_id)
                        uid2ans[uid] = ans

                for k, v in results.items():
                    if k in epoch_results:
                        epoch_results[k] += v.item()

                if self.verbose:
                    desc_str = f'Valid Epoch {epoch} | '

                    # if self.args.task_qa:
                    #     loss_meter.update(results['qa_loss'].item())
                    if self.args.task_qa:
                        loss_meter = loss_meters[-1]
                        loss_meter.update(results['qa_loss'].item())
                        loss_counts[-1] += 1

                    for i, (loss_name, loss_meter) in enumerate(zip(LOSSES_NAME, loss_meters)):
                        if task_dict[loss_name] == task:
                            if task == 'vis_mask':
                                if loss_name == 'Mask_Obj':
                                    loss_meter.update(results['obj_loss'].item())
                                elif loss_name == 'Mask_Attr':
                                    loss_meter.update(results['attr_loss'].item())
                                elif loss_name == 'Mask_Feat':
                                    loss_meter.update(results['feat_loss'].item())
                            elif task == 'word_mask':
                                loss_meter.update(results['lm_loss'].item())
                            elif task == 'matched':
                                loss_meter.update(results['matched_loss'].item())
                            # elif task == 'qa':
                            #     loss_meter.update(results['qa_loss'].item())
                            loss_counts[i] += 1
                        if len(loss_meter) > 0:
                            loss_count = loss_counts[i]
                            if loss_name in ['Mask_LM', 'Matched', 'Mask_Obj', 'Mask_Attr', 'Mask_Feat', 'QA']:
                                desc_str += f' {loss_name} ({loss_count}) {loss_meter.val:.2f}'
                            else:
                                desc_str += f' {loss_name} {loss_meter.val:.2f}'

                    pbar.set_description(desc_str)
                    pbar.update(1)
                dist.barrier()

            if self.verbose:
                pbar.close()
            dist.barrier()

            if not self.args.task_qa:
                uid2ans = None

            return epoch_results, uid2ans

    def save(self, name):
        torch.save(self.model.state_dict(),
                   os.path.join(self.args.output, "%s_LXRT.pth" % name))

    def load(self, path, loc=None, verbose=False):
        print("Load BERT extractor from %s" % path)
        if loc is None:
            state_dict = torch.load("%s_LXRT.pth" % path)
        else:
            state_dict = torch.load("%s_LXRT.pth" % path, map_location=loc)
        result = self.model.load_state_dict(state_dict, strict=False)
        if verbose:
            print(result)

    def load_lxmert(self, path, loc=None, verbose=False):
        if verbose:
            print("Load LXMERT model from %s" % path)
        if loc is None:
            state_dict = torch.load("%s_LXRT.pth" % path)
        else:
            state_dict = torch.load("%s_LXRT.pth" % path, map_location=loc)

        # Do not load any answer head
        for key in list(state_dict.keys()):
            if 'answer' in key:
                state_dict.pop(key)

        # Change Multi GPU to single GPU
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                new_state_dict[key[len("module."):]] = value
        state_dict = new_state_dict
        if verbose:
            load_keys = set(state_dict.keys())
            model_keys = set(self.model.module.state_dict().keys())
            print()
            print("Keys in loaded but not in model:")
            for key in sorted(load_keys.difference(model_keys)):
                print(key)
            print()
            print("Keys in model but not in loaded:")
            for key in sorted(model_keys.difference(load_keys)):
                print(key)
            print()

        self.model.module.load_state_dict(state_dict, strict=False)


class LossMeter(object):
    def __init__(self, maxlen=100):
        """Computes and stores the running average"""
        self.vals = collections.deque([], maxlen=maxlen)

    def __len__(self):
        return len(self.vals)

    def update(self, new_val):
        self.vals.append(new_val)

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
        shutil.copytree(src_dir, log_src_dir)
        # print('Source files logged at', log_src_dir)
        logger.info('Source files logged at' + str(log_src_dir))

    else:
        logger = None

    # transform = transforms.Compose([
    #     transforms.Resize((args.img_size,
    #                        args.img_size)),
    #     # transforms.RandomHorizontalFlip()
    # ])
    transform = None

    data_out = ['box', 'sent']

    if args.task_matched:
        data_out += ['matched']

    # data_out = ['img', 'box', 'sent']
    if args.clustering or args.target_cluster:
        # collate_fn = image_cluster_id_collate_fn
        data_out += ['cluster_id']
    if args.word_mask_predict:
        data_out += ['word_mask_idx']
    if args.vis_mask_predict:
        data_out += ['vis_mask_idx']
    if args.feed_exact_feat or args.target_exact_feat:
        data_out += ['feat']
    if args.target_prob:
        if args.grid_model:
            data_out += ['prob']
        else:
            data_out += ['obj_prob']
            data_out += ['attr_prob']
    if args.target_obj_id:
        data_out += ['obj_id']

    if args.task_qa:
        data_out += ['ans']

    args.data_out = data_out

    train_loader = get_loader(
        args,
        # 'mscoco_minival', mode='train', batch_size=args.batch_size,
        split=args.train, mode='train', batch_size=args.batch_size,
        distributed=args.distributed, gpu=args.gpu,
        workers=args.num_workers, transform=transform,
        topk=args.train_topk, data_out=data_out)

    val_loader = get_loader(
        args,
        split=args.valid, mode='val', batch_size=args.batch_size,
        distributed=args.distributed, gpu=args.gpu,
        workers=args.num_workers, transform=transform,
        topk=args.valid_topk, data_out=data_out)

    trainer = LXMERT(args, train_loader, val_loader, logger, train=True)
    # trainer = LXMERT(args, train_loader, val_loader, logger, train=False)

    # import ipdb
    # ipdb.set_trace()

    trainer.train()


if __name__ == "__main__":
    cudnn.benchmark = True
    # args = parse_args()

    if args.vis_mask_predict:
        args.obj_mask_rate = 'uniform'

    if args.word_mask_predict:
        args.word_mask_rate = 'uniform'

    args.n_grids = args.grid_size ** 2

    print(args)

    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node

    LOSSES_NAME = []

    if args.task_mask_lm:
        LOSSES_NAME.append('Mask_LM')
    if args.task_matched:
        LOSSES_NAME.append('Matched')
    if 'obj' in args.visual_losses.split(','):
        LOSSES_NAME.append('Mask_Obj')
    if 'attr' in args.visual_losses.split(','):
        LOSSES_NAME.append('Mask_Attr')
    if 'feat' in args.visual_losses.split(','):
        LOSSES_NAME.append('Mask_Feat')
    if 'img' in args.visual_losses.split(','):
        LOSSES_NAME.append('Mask_Img')
    if args.task_qa:
        LOSSES_NAME.append('QA')

    MASK_MODALITY = []
    if args.task_obj_predict:
        MASK_MODALITY.append('vis_mask')
    if args.task_mask_lm:
        MASK_MODALITY.append('word_mask')
    if args.task_matched:
        MASK_MODALITY.append('matched')
    # if args.task_qa:
    #     MASK_MODALITY.append('qa')

    print(LOSSES_NAME)
    print(MASK_MODALITY)
    args.LOSSES_NAME = LOSSES_NAME
    args.MASK_MODALITY = MASK_MODALITY

    if args.grid_model:
        comment = f'Grid{args.grid_size}'
    else:
        comment = f'Box{args.n_boxes}'
    if args.clustering:
        comment += '_cluster'
    if args.backbone == 'uniter':
        comment += '_uniter'
    if 'mscoco_train' in args.train:
        if 'vgnococo' in args.train:
            comment += '_COCOVG'
        else:
            comment += '_COCO'
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
    if args.task_qa:
        comment += f'_QA'
    if args.target_cluster:
        comment += '_targetcluster'
    if args.clustering:
        comment += f'_Vocab{args.n_centroids}'

    comment += f'_{args.encoder}'
    if args.im_ratio == 'original':
        comment += f'_imratio_{args.im_ratio}'
    else:
        comment += f'_imsize{args.resize_input_size}'

    if args.vis_mask_COCO_only:
        comment += '_COCOvismask'
    elif args.vis_mask_COCOVG_only:
        comment += '_COCOVGvismask'

    if args.comment != '':
        comment += f'_{args.comment}'

    from datetime import datetime
    current_time = datetime.now().strftime('%b%d_%H-%M')
    project_dir = Path(__file__).resolve().parent.parent.parent
    log_dir = project_dir.joinpath('runs')
    log_str = 'pretrain'
    # if args.clustering:
    #     log_str += '_cluster'
    log_dir = log_dir.joinpath(log_str, current_time + f'_GPU{args.world_size}_' + comment)
    args.log_dir = log_dir
    log_dir.mkdir(exist_ok=True, parents=True)

    if args.distributed:
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(args,))
    else:
        main_worker(0, args)
