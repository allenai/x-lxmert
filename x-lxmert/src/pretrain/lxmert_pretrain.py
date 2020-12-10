# coding=utf-8

import collections
import os
import random
from pathlib import Path
import logging
import shutil
from packaging import version

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn

from transformers import LxmertTokenizer

_use_native_amp = False
_use_apex = False

# Check if Pytorch version >= 1.6 to switch between Native AMP and Apex
if version.parse(torch.__version__) < version.parse("1.6"):
    from transormers.file_utils import is_apex_available
    if is_apex_available():
        from apex import amp
    _use_apex = True
else:
    _use_native_amp = True
    from torch.cuda.amp import autocast


from param import parse_args
from lxrt.modeling import XLxmertForPretraining
from pretrain.lxmert_data import get_loader
from utils import load_state_dict, LossMeter, count_parameters, reduce_dict, set_global_logging_level

set_global_logging_level(logging.ERROR, ["transformers"])
cudnn.benchmark = True

class Trainer:
    def __init__(self, args, train_loader=None, val_loader=None, logger=None, train=True):
        super().__init__()

        self.args = args
        self.max_text_length = args.max_text_length

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.logger = logger

        # Build model
        self.model = XLxmertForPretraining.from_pretrained(
            "bert-base-uncased",
            num_clusters=args.num_clusters
        )

        self.tokenizer = LxmertTokenizer.from_pretrained('unc-nlp/lxmert-base-uncased')

        self.verbose = True
        if self.args.distributed:
            if self.args.gpu != 0:
                self.verbose = False

        if args.clustering:
            self.datasets_dir = Path(self.args.datasets_dir)
            clustering_dir = self.datasets_dir.joinpath('clustering')
            centroid_path = clustering_dir.joinpath(
                f'{args.encoder}_{args.cluster_src}_centroids{args.n_centroids}_iter{args.n_iter}_d{args.feat_dim}_grid{args.grid_size}.npy')
            centroids = np.load(centroid_path)

            self.model.set_visual_embedding(centroids)

        # Load pre-trained weights
        self.start_epoch = None
        if args.load is not None:
            path = args.load + '_LXRT.pth'
            self.load(path, verbose=self.verbose)

        # GPU Options
        print(f'Model Launching at GPU {self.args.gpu}')

        from time import time
        start = time()
        self.model = self.model.to(args.gpu)

        # Optimizer
        if train:
            self.optim, self.lr_scheduler = self.create_optimizer_and_scheduler()

            if self.args.fp16 and _use_native_amp:
                self.scaler = torch.cuda.amp.GradScaler()
            elif _use_apex:
                self.model, self.optim = amp.initialize(
                    self.model, self.optim, opt_level='O1', verbosity=self.verbose)

        if args.multiGPU:
            assert args.distributed
            self.model = DDP(self.model, device_ids=[args.gpu],
                                find_unused_parameters=True
                                )
        if args.gpu == 0:
            print(f'It took {time() - start:.1f}s')

    def create_optimizer_and_scheduler(self):
        if self.verbose:
            print('Building Optimizer')

        from transformers.optimization import AdamW, get_linear_schedule_with_warmup
        batch_per_epoch = len(self.train_loader)
        t_total = int(batch_per_epoch * self.args.epochs)
        warmup_ratio = self.args.warmp_ratio
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
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        optim = AdamW(optimizer_grouped_parameters, self.args.lr)
        lr_scheduler = get_linear_schedule_with_warmup(
            optim, warmup_iters, t_total)

        return optim, lr_scheduler

    def forward(self, batch, task):

        device = torch.cuda.current_device()

        if self.args.clustering:
            visual_feats = None
            cluster_ids = batch['cluster_id'].to(device)
        else:
            visual_feats = batch['vis_feats'].to(device)
            cluster_ids = None

        visual_pos = batch['box_position'].to(device)

        vis_mask = batch['vis_mask'].to(device).bool()

        label_dict = {}
        if task == 'word_mask':
            label_dict['word_labels'] = batch['word_label'].to(device)
        elif task == 'vis_mask':
            if 'obj' in self.args.visual_losses:
                if self.args.clustering:
                    obj_labels = cluster_ids.detach()
                    obj_labels[~vis_mask] = -100
                    label_dict['obj_labels'] = obj_labels
                elif self.args.target_obj_id:
                    obj_labels = batch['obj_ids'].to(device)
                    obj_labels[~vis_mask] = -100
                    label_dict['obj_labels'] = obj_labels

            if 'attr' in self.args.visual_losses:
                attr_labels = batch['attr_label'].to(device)
                attr_labels[~vis_mask] = -100
                label_dict['attr_labels'] = attr_labels

            if 'feat' in self.args.visual_losses:
                vis_feats = batch['vis_feats'].detach().to(device)
                label_dict['feat_labels'] = vis_feats
        elif task == 'matched':
            matched_label = batch['matched_label'].to(device)
            label_dict['matched_labels'] = matched_label
        # elif task == 'qa':
        if self.args.task_qa:
            qa_label = batch['qa_label'].to(device)
            if task == 'matched':
                flipped = (matched_label == 0)
                qa_label.masked_fill_(flipped, -100)
            label_dict['qa_labels'] = qa_label


        if task == 'word_mask':
            word_id = batch['masked_word_id'].to(device)
        elif task == 'matched':
            word_id = batch['other_word_id'].to(device)
        elif task == 'vis_mask':
            word_id = batch['word_id'].to(device)

        # word_attention_mask = word_id > 0
        token_type_ids = torch.zeros_like(word_id)

        out_dict = self.model(
            input_ids=word_id,
            visual_feats=visual_feats,
            visual_pos=visual_pos,
            attention_mask=word_id > 0,
            visual_attention_mask=None,

            cluster_ids=cluster_ids,
            vis_mask=vis_mask,

            token_type_ids=token_type_ids,
            # labels=None,
            # obj_labels=None,
            # matched_label=None,
            # ans=None,

            return_dict=True,

            label_dict=label_dict,

            task=task,
            )

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

                # with torch.autograd.set_detect_anomaly(True):
                results = self.forward(batch, task)

                if self.args.fp16 and _use_native_amp:
                    with autocast():
                        results = self.model(batch, task)
                else:
                    results = self.model(batch, task)

                if task == 'vis_mask':
                    if 'Mask_Obj' in LOSSES_NAME:
                        epoch_results['obj_loss_count'] += 1
                    if 'Mask_Feat' in LOSSES_NAME:
                        epoch_results['feat_loss_count'] += 1
                    if 'Mask_Attr' in LOSSES_NAME:
                        epoch_results['attr_loss_count'] += 1
                    epoch_results['vis_loss_count'] += 1
                elif task == 'word_mask':
                    epoch_results['lm_loss_count'] += 1
                elif task == 'matched':
                    epoch_results['matched_loss_count'] += 1

                if self.args.task_qa:
                    epoch_results['qa_loss_count'] += 1
                    qa_pred = results['qa_pred']
                    for uid, ans_id in zip(batch['uid'], qa_pred.cpu().numpy()):
                        ans = self.train_loader.dataset.answer_table.id2ans(ans_id)
                        uid2ans[uid] = ans

                loss = results['total_loss']

                #===== Update =====#
                if self.args.fp16 and _use_native_amp:
                    self.scaler.scale(loss).backward()
                elif self.args.fp16 and _use_apex:
                    with amp.scale_loss(loss, self.optim) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                loss = loss.detach()

                # Update Parameters
                if self.args.clip_grad_norm > 0:
                    if self.args.fp16 and _use_native_amp:
                        self.scaler.unscale_(self.optim)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.args.clip_grad_norm)
                    elif self.args.fp16 and _use_apex:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(
                            self.optim), self.args.clip_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.args.clip_grad_norm)

                if self.args.fp16 and _use_native_amp:
                    self.scaler.step(self.optim)
                    self.scaler.update()
                else:
                    self.optim.step()

                if self.lr_scheduler:
                    self.lr_scheduler.step()
                for param in self.model.parameters():
                    param.grad = None

                global_step += 1
                #====================#

                try:
                    lr = self.scheduler.get_last_lr()[0]
                except AttributeError:
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

                    # if update:
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

    def load(self, path, loc='cpu', verbose=False):
        state_dict = load_state_dict(path, loc)
        results = self.model.load_state_dict(state_dict, strict=False)
        if verbose:
            print('Loaded from ', path)
            print(results)
        self.start_epoch = int(args.load.split('Epoch')[-1])


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

    data_out = ['box', 'sent']

    if args.task_matched:
        data_out += ['matched']
    if args.clustering or args.target_cluster:
        data_out += ['cluster_id']
    if args.word_mask_predict:
        data_out += ['word_mask_idx']
    if args.vis_mask_predict:
        data_out += ['vis_mask_idx']
    if args.feed_exact_feat or args.target_exact_feat:
        data_out += ['feat']
    if args.target_obj_id:
        data_out += ['obj_id']

    if args.task_qa:
        data_out += ['ans']

    args.data_out = data_out

    train_loader = get_loader(
        args,
        split=args.train, mode='train', batch_size=args.batch_size,
        distributed=args.distributed, gpu=args.gpu,
        workers=args.num_workers,
        topk=args.train_topk, data_out=data_out)

    val_loader = get_loader(
        args,
        split=args.valid, mode='val', batch_size=args.batch_size,
        distributed=args.distributed, gpu=args.gpu,
        workers=args.num_workers,
        topk=args.valid_topk, data_out=data_out)

    trainer = Trainer(args, train_loader, val_loader, logger, train=True)
    trainer.train()


if __name__ == "__main__":
    args = parse_args()

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
