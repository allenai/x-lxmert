# coding=utf-8
# Copyleft 2019 project LXRT.

import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import collections
from pathlib import Path

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
# from torch.utils.data.dataloader import DataLoader
import logging
import shutil
from pprint import pprint

# from param import args
from param import parse_args
from tasks.nlvr2_model import NLVR2Model
# from tasks.nlvr2_data import NLVR2Dataset, NLVR2TorchDataset, NLVR2Evaluator
from tasks.nlvr2_data import get_loader

# feat_dim = args.feat_dim
# n_class = args.n_class
# n_centroids = args.n_centroids
# n_iter = args.n_iter
# cluster_src = args.cluster_src

#
# DataTuple = collections.namedtuple("DataTuple", 'dataset loader evaluator')
#
#
# def get_tuple(args, splits: str, bs: int, shuffle=False, drop_last=False) -> DataTuple:
#     dset = NLVR2Dataset(splits)
#     tset = NLVR2TorchDataset(args, dset)
#     evaluator = NLVR2Evaluator(dset)
#     data_loader = DataLoader(
#         tset, batch_size=bs,
#         shuffle=shuffle, num_workers=args.num_workers,
#         drop_last=drop_last, pin_memory=True
#     )
#
#     return DataTuple(dataset=dset, loader=data_loader, evaluator=evaluator)

def load_state_dict(state_dict_path, loc='cpu'):
    state_dict = torch.load(state_dict_path, map_location=loc)
    # Change Multi GPU to single GPU
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            new_state_dict[key[len("module."):]] = value
        else:
            new_state_dict[key] = value
    return new_state_dict

class NLVR2:
    def __init__(self, args, train_loader=None, val_loader=None, logger=None, train=True):
        self.args = args
        self.max_text_length = args.max_text_length

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.logger = logger

        # self.train_tuple = get_tuple(args,
        #                              args.train, bs=args.batch_size, shuffle=True, drop_last=True
        #                              )
        # if args.valid != "":
        #     valid_bsize = 2048 if args.multiGPU else 512
        #     self.valid_tuple = get_tuple(args,
        #                                  args.valid, bs=valid_bsize,
        #                                  shuffle=False, drop_last=False
        #                                  )
        # else:
        #     self.valid_tuple = None

        from lxrt.modeling import VISUAL_CONFIG
        VISUAL_CONFIG.l_layers = args.llayers
        VISUAL_CONFIG.x_layers = args.xlayers
        VISUAL_CONFIG.r_layers = args.rlayers
        VISUAL_CONFIG.visual_feat_dim = args.feat_dim

        self.model = NLVR2Model.from_pretrained(
            "bert-base-uncased",
            args=args,
        )

        self.verbose = True
        if self.args.distributed:
            if self.args.gpu != 0:
                self.verbose = False

        # GPU Options
        print(f'Model Launching at GPU {self.args.gpu}')
        from time import time
        start = time()
        self.model.cuda(args.gpu)

        # Load Checkpoint
        self.start_epoch = None
        if args.load is not None:
            if args.start_epoch is not None:
                self.start_epoch = args.start_epoch
            # loc = 'cuda:{}'.format(args.gpu)
            # loc = 'cpu'
            # self.load(args.load, loc=loc)
            ckpt = args.load + '.pth'
            state_dict = load_state_dict(ckpt, 'cpu')
            # results = self.model.lxrt_encoder.load_state_dict(state_dict, strict=False)
            results = self.model.load_state_dict(state_dict, strict=False)
            if self.verbose:
                print('NLVR model loaded from ', ckpt)
                pprint(results)

        # Load pre-trained weights
        elif args.load_lxmert is not None:
            # loc = 'cuda:{}'.format(args.gpu)
            # loc = 'cpu'
            # self.model.lxrt_encoder.load(
            #     args.load_lxmert, loc=loc, verbose=self.verbose)
            lxmert_ckpt = args.load_lxmert + '_LXRT.pth'
            state_dict = load_state_dict(lxmert_ckpt, 'cpu')
            # results = self.model.lxrt_encoder.load_state_dict(state_dict, strict=False)
            results = self.model.load_state_dict(state_dict, strict=False)
            if self.verbose:
                print('LXRT encoder loaded from ', lxmert_ckpt)
                pprint(results)

        if not args.test:
            # Losses and optimizer
            self.mce_loss = nn.CrossEntropyLoss(ignore_index=-1)
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
                self.optim = BertAdam(list(self.model.parameters()),
                                      lr=args.lr,
                                      warmup=warmup_ratio,
                                      t_total=t_total)
            elif 'adamw' in args.optim:
                from transformers.optimization import AdamW, get_linear_schedule_with_warmup
                batch_per_epoch = len(train_loader)
                t_total = int(batch_per_epoch * args.epochs)
                warmup_ratio = 0.1
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
                self.scheduler = get_linear_schedule_with_warmup(self.optim, warmup_iters, t_total)
            else:
                self.optim = args.optimizer(list(self.model.parameters()), args.lr)

            if args.mixed_precision:
                self.model, self.optim = amp.initialize(
                    self.model, self.optim, opt_level='O1', verbosity=self.verbose)

            if args.multiGPU:
                if args.distributed:
                    self.model = DDP(self.model, device_ids=[args.gpu],
                                     find_unused_parameters=True
                                     )
                else:
                    self.model = nn.DataParallel(self.model)

        if args.gpu == 0:
            print(f'It took {time() - start:.1f}s')

        self.output = args.output
        os.makedirs(self.output, exist_ok=True)

        cudnn.benchmark = True

    def train(self):
        if self.verbose:
            loss_meter = LossMeter()
            best_eval_loss = 9595.
            best_valid = 0.

            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=self.args.log_dir)

            hparam_dict = {}
            for k, v in self.args.__dict__.items():
                if type(v) in [int, float, str, bool, torch.Tensor]:
                    hparam_dict[k] = v
            metric_dict = {}

            self.writer.add_hparams(hparam_dict, metric_dict)

        if self.args.distributed:
            dist.barrier()

        self.optim.zero_grad()

        for epoch in range(self.args.epochs):
            self.model.train()
            if self.args.distributed:
                self.train_loader.sampler.set_epoch(epoch)
            if self.verbose:
                pbar = tqdm(total=len(self.train_loader), ncols=150)

                results = np.zeros(4, dtype=int)
                quesid2ans = {}

            for step_i, batch in enumerate(self.train_loader):
                if self.args.clustering:
                    cluster_ids = batch['cluster_ids'].cuda()
                    if type(self.model) in [DDP, nn.DataParallel]:
                        vis_feats = self.model.module.vis_emb(cluster_ids)
                    else:
                        vis_feats = self.model.vis_emb(cluster_ids)
                else:
                    vis_feats = batch['vis_feats'].cuda()
                boxes = batch['boxes'].cuda()

                ques_id = batch['question_ids']
                B = len(ques_id)

                input_ids = batch['word_ids'].cuda()
                input_ids = input_ids.unsqueeze(1).repeat(1, 2, 1).view(B * 2, -1)
                token_type_ids = torch.zeros_like(input_ids)
                word_attention_mask = input_ids > 0

                label = batch['labels'].cuda()

                visn_feats = (vis_feats, boxes)
                sent_feats = (input_ids, token_type_ids, word_attention_mask)

                logit = self.model(visn_feats, sent_feats)
                loss = self.mce_loss(logit, label)

                loss.backward()

                update = True
                if self.args.update_freq > 1:
                    if step_i == 0:
                        update = False
                    elif step_i % self.args.update_freq == 0 or step_i == len(self.train_loader) - 1:
                        update = True
                    else:
                        update = False

                if update:
                    # nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                    if not self.args.no_clip_grad:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)

                    self.optim.step()
                    if self.args.optim == 'adamw':
                        self.scheduler.step()
                    self.optim.zero_grad()

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
                    loss_meter.update(loss.item())
                    desc_str = f'Epoch {epoch} | LR {lr:.6f} | '
                    desc_str += f'Loss {loss_meter.val:4f} |'

                    score, predict = logit.max(1)
                    predict = predict.cpu().numpy()
                    label = label.cpu().numpy()

                    for qid, pred in zip(ques_id, predict):
                        quesid2ans[qid] = pred

                    results[0] += sum((label == 1) & (predict == 1))
                    results[1] += sum((label == 1) & (predict == 0))
                    results[2] += sum((label == 0) & (predict == 1))
                    results[3] += sum((label == 0) & (predict == 0))
                    n_total = sum(results)

                    desc_str += f' TP {results[0]} ({results[0]/n_total*100:.1f}%)'
                    desc_str += f' FN {results[1]} ({results[1]/n_total*100:.1f}%)'
                    desc_str += f' FP {results[2]} ({results[2]/n_total*100:.1f}%)'
                    desc_str += f' TN {results[3]} ({results[3]/n_total*100:.1f}%)'

                    pbar.set_description(desc_str)
                    pbar.update(1)

                if self.args.distributed:
                    dist.barrier()

            if self.verbose:
                pbar.close()
                score = self.train_loader.evaluator.evaluate(quesid2ans) * 100.
                log_str = "\nEpoch %d: Train %0.2f" % (epoch, score)

                if not self.args.dry:
                    self.writer.add_scalar(f'NLVR/Train/score', score, epoch)

                # Validation
                valid_score = self.evaluate(self.val_loader) * 100.
                if valid_score > best_valid:
                    best_valid = valid_score
                    self.save("BEST")

                log_str += "\nEpoch %d: Valid %0.2f" % (epoch, valid_score)
                log_str += "\nEpoch %d: Best %0.2f\n" % (epoch, best_valid)

                if not self.args.dry:
                    self.writer.add_scalar(
                        f'NLVR/Valid/score', valid_score, epoch)

                print(log_str)
                self.logger.info(log_str)

            if self.args.distributed:
                dist.barrier()

        if self.verbose:
            self.save("LAST")

    def predict(self, loader, dump_path=None, test=False):
        self.model.eval()
        with torch.no_grad():
            quesid2ans = {}

            if test:
                pbar = tqdm(ncols=100, total=len(loader))
            else:
                # Label / Pred
                results = np.zeros(4, dtype=int)

            for i, batch in enumerate(loader):
                if self.args.clustering:
                    cluster_ids = batch['cluster_ids'].cuda()
                    if type(self.model) in [DDP, nn.DataParallel]:
                        vis_feats = self.model.module.vis_emb(cluster_ids)
                    else:
                        vis_feats = self.model.vis_emb(cluster_ids)
                else:
                    vis_feats = batch['vis_feats'].cuda()
                boxes = batch['boxes'].cuda()

                ques_id = batch['question_ids']
                B = len(ques_id)

                input_ids = batch['word_ids'].cuda()
                input_ids = input_ids.unsqueeze(1).repeat(1, 2, 1).view(B * 2, -1)
                token_type_ids = torch.zeros_like(input_ids)
                word_attention_mask = input_ids > 0

                if not test:
                    label = batch['labels']# .cuda()

                visn_feats = (vis_feats, boxes)
                sent_feats = (input_ids, token_type_ids, word_attention_mask)

                logit = self.model(visn_feats, sent_feats)

                score, predict = logit.max(1)

                predict = predict.cpu().numpy()

                for qid, pred in zip(ques_id, predict):
                        quesid2ans[qid] = pred
                if test:
                    pbar.update(1)
                else:
                    label = label.numpy()
                    results[0] += sum((label == 1) & (predict == 1))
                    results[1] += sum((label == 1) & (predict == 0))
                    results[2] += sum((label == 0) & (predict == 1))
                    results[3] += sum((label == 0) & (predict == 0))

            if not test:
                n_total = sum(results)

                desc_str = 'Valid'
                desc_str += f' TP {results[0]} ({results[0]/n_total*100:.1f}%)'
                desc_str += f' FN {results[1]} ({results[1]/n_total*100:.1f}%)'
                desc_str += f' FP {results[2]} ({results[2]/n_total*100:.1f}%)'
                desc_str += f' TN {results[3]} ({results[3]/n_total*100:.1f}%)'
                print(desc_str)
                self.logger.info(desc_str)

        if test:
            pbar.close()

        if dump_path is not None:
            loader.evaluator.dump_result(quesid2ans, dump_path)
        return quesid2ans

    def evaluate(self, loader, dump_path=None):
        evaluator = loader.evaluator
        quesid2ans = self.predict(loader, dump_path)
        return evaluator.evaluate(quesid2ans)

    def save(self, name):
        torch.save(self.model.state_dict(),
                   os.path.join(self.output, "%s.pth" % name))

    def load(self, path):
        print("Load model from %s" % path)
        state_dict = torch.load("%s.pth" % path)
        self.model.load_state_dict(state_dict)


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
    if gpu == 0 and (not args.test):
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
        print('Source files logged at', log_src_dir)
        logger.info('Source files logged at' + str(log_src_dir))

    else:
        logger = None

    if not args.test:
        transform = None
        train_loader = get_loader(
            args,
            # 'mscoco_minival', mode='train', batch_size=args.batch_size,
            split=args.train, mode='train', batch_size=args.batch_size,
            distributed=args.distributed, gpu=args.gpu,
            workers=args.num_workers, transform=transform,
            # topk=args.train_topk, data_out=data_out)
        )

        val_loader = get_loader(
            args,
            split=args.valid, mode='val', batch_size=args.batch_size,
            distributed=args.distributed, gpu=args.gpu,
            workers=4, transform=transform,
            # topk=args.valid_topk, data_out=data_out)
        )

        trainer = NLVR2(args, train_loader, val_loader, logger, train=True)
        trainer.train()
    else:
        print('Evaluation on test-P split')
        train_loader = None
        transform = None
        test_loader = get_loader(
            args,
            split='test', mode='val', batch_size=args.batch_size,
            distributed=False, gpu=args.gpu,
            workers=4, transform=transform,
            # topk=args.valid_topk, data_out=data_out)
        )

        trainer = NLVR2(args, train_loader, test_loader, logger, train=False)
        dump_path = 'Test_submission/nlvr2/' + args.comment + '_test_P.csv'
        print('dump_path is', dump_path)
        trainer.predict(test_loader, dump_path=dump_path, test=True)


if __name__ == "__main__":
    cudnn.benchmark = True
    args = parse_args()
    print(args)
    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node
    args.n_grids = args.grid_size ** 2

    if args.grid_model:
        comment = f'Grid{args.grid_size}'
    else:
        comment = f'Box{args.n_boxes}'
    if args.clustering:
        comment += '_cluster'
    comment += f'_{args.backbone}'
    comment += f'_{args.encoder}'

    if args.im_ratio == 'original':
        comment += f'_imratio{args.im_ratio}'
    else:
        comment+= f'_imsize{args.resize_input_size}'
    comment += f'_dim{args.feat_dim}'

    if args.load_lxmert is not None:
        comment += f'_load_{args.load_lxmert}'

    if args.comment != '':
        comment += f'_{args.comment}'

    args.comment = comment

    if not args.test:
        from datetime import datetime
        current_time = datetime.now().strftime('%b%d_%H-%M')
        project_dir = Path(__file__).resolve().parent.parent.parent
        log_dir = project_dir.joinpath('runs')
        log_dir = log_dir.joinpath(
            'NLVR2_finetune', current_time + '_' + comment)
        args.log_dir = log_dir
        log_dir.mkdir(exist_ok=True, parents=True)
        print('logging at', log_dir)

    # nlvr2.train(nlvr2.train_tuple, nlvr2.valid_tuple)
    if args.distributed:
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(args,))
    else:
        main_worker(0, args)
