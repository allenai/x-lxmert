# coding=utf-8
# Copyleft 2019 project LXRT.

import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import collections
from pathlib import Path

from apex import amp

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
# from torch.utils.data.dataloader import DataLoader
import logging
import shutil
from pprint import pprint

from param import args

from pretrain.qa_answer_table import load_lxmert_qa
from tasks.gqa_model import GQAModel
from tasks.gqa_data import get_loader
# from tasks.gqa_data import GQADataset, GQATorchDataset, GQAEvaluator

from utils import load_state_dict, LossMeter


class GQA:
    def __init__(self, args, train_loader=None, val_loader=None, logger=None, num_answers=0, train=True):
        self.args = args
        self.max_text_length = args.max_text_length
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_answers = num_answers
        self.logger = logger

        # Model
        # self.model = VQAModel(args, self.train_loader.dataset.raw_dataset.num_answers)
        # from lxrt.entry import set_visual_config
        # set_visual_config(args)
        from lxrt.modeling import VISUAL_CONFIG
        VISUAL_CONFIG.l_layers = args.llayers
        VISUAL_CONFIG.x_layers = args.xlayers
        VISUAL_CONFIG.r_layers = args.rlayers
        VISUAL_CONFIG.visual_feat_dim = args.feat_dim

        self.model = GQAModel.from_pretrained(
            "bert-base-uncased",
            args=args,
            num_answers=self.num_answers
        )

        self.verbose = True
        if self.args.distributed:
            if self.args.gpu != 0:
                self.verbose = False

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
                print('GQA model loaded from ', ckpt)
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

        elif args.load_lxmert_qa is not None:
            load_lxmert_qa(args.load_lxmert_qa, self.model,
                           label2ans=self.train_loader.dataset.raw_dataset.label2ans)

        # GPU Options
        print(f'Model Launching at GPU {self.args.gpu}')
        from time import time
        start = time()
        self.model.cuda(args.gpu)

        if not args.test:
            # Loss and Optimizer
            self.bce_loss = nn.BCEWithLogitsLoss()
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
                self.scheduler = get_linear_schedule_with_warmup(
                    self.optim, warmup_iters, t_total)
            else:
                self.optim = args.optimizer(
                    list(self.model.parameters()), args.lr)

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

        # Output Directory
        self.output = args.output
        os.makedirs(self.output, exist_ok=True)

        cudnn.benchmark = True

    def train(self):
        if self.verbose:
            loss_meter = LossMeter()
            best_eval_loss = 9595.
            quesid2ans = {}
            best_valid = 0.
            print("Valid Oracle: %0.2f" %
                  (self.oracle_score(self.val_loader) * 100))

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
            if self.start_epoch is not None:
                epoch += self.start_epoch
            self.model.train()
            if self.args.distributed:
                self.train_loader.sampler.set_epoch(epoch)
            if self.verbose:
                pbar = tqdm(total=len(self.train_loader), ncols=150)

            quesid2ans = {}
            for step_i, batch in enumerate(self.train_loader):
                update = True
                if self.args.update_freq > 1:
                    if step_i == 0:
                        update = False
                    elif step_i % self.args.update_freq == 0 or step_i == len(self.train_loader) - 1:
                        update = True
                    else:
                        update = False

                if self.args.clustering:
                    cluster_ids = batch['cluster_ids'].cuda()
                    # [B, n_grids, code_dim]
                    # if self.args.distributed:
                    if type(self.model) in [DDP, nn.DataParallel]:
                        vis_feats = self.model.module.vis_emb(cluster_ids)
                    else:
                        vis_feats = self.model.vis_emb(cluster_ids)
                else:
                    vis_feats = batch['vis_feats'].cuda()
                boxes = batch['boxes'].cuda()

                input_ids = batch['word_ids'].cuda()
                token_type_ids = torch.zeros_like(input_ids)
                word_attention_mask = input_ids > 0
                target = batch['targets'].cuda()

                ques_id = batch['question_ids']

                B = len(batch['word_ids'])

                visn_feats = (vis_feats, boxes)
                sent_feats = (input_ids, token_type_ids, word_attention_mask)

                logit = self.model(visn_feats, sent_feats)

                assert logit.size() == target.size()
                assert logit.size() == (B, self.num_answers)

                loss = self.bce_loss(logit, target)

                loss.backward()

                if update:
                    if not self.args.no_clip_grad:
                        # if self.args.distributed:
                        #     nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
                        # else:
                        # nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                        nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.args.clip_grad_norm)

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
                    target = target.cpu().numpy()

                    for qid, pred in zip(ques_id, predict):
                        pred_ans = self.train_loader.dataset.raw_dataset.label2ans[pred]
                        quesid2ans[qid] = pred_ans

                    pbar.set_description(desc_str)
                    pbar.update(1)

                if self.args.distributed:
                    dist.barrier()

                # score, label = logit.max(1)
                # for qid, l in zip(ques_id, label.cpu().numpy()):
                #     ans = dset.label2ans[l]
                #     quesid2ans[qid.item()] = ans

            if self.verbose:
                pbar.close()
                score = self.train_loader.evaluator.evaluate(quesid2ans) * 100.
                log_str = "\nEpoch %d: Train %0.2f" % (epoch, score)

                if not self.args.dry:
                    self.writer.add_scalar(f'GQA/Train/score', score, epoch)

                # Validation
                valid_score = self.evaluate(self.val_loader) * 100.
                if valid_score > best_valid:
                    best_valid = valid_score
                    self.save("BEST")

                log_str += "\nEpoch %d: Valid %0.2f" % (epoch, valid_score)
                log_str += "\nEpoch %d: Best %0.2f\n" % (epoch, best_valid)

                if not self.args.dry:
                    self.writer.add_scalar(
                        f'GQA/Valid/score', valid_score, epoch)

                print(log_str)
                self.logger.info(log_str)

            if self.args.distributed:
                dist.barrier()

        if self.verbose:
            self.save("LAST")

    def predict(self, loader, dump_path=None):
        """
        Predict the answers to questions in a data split.

        :param eval_tuple: The data tuple to be evaluated.
        :param dump: The path of saved file to dump results.
        :return: A dict of question_id to answer.
        """
        self.model.eval()
        with torch.no_grad():
            quesid2ans = {}
            for i, batch in enumerate(tqdm(loader, ncols=100, desc="Prediction")):

                if self.args.clustering:
                    cluster_ids = batch['cluster_ids'].cuda()
                    # [B, n_grids, code_dim]
                    if type(self.model) in [DDP, nn.DataParallel]:
                        vis_feats = self.model.module.vis_emb(cluster_ids)
                    else:
                        vis_feats = self.model.vis_emb(cluster_ids)
                else:
                    vis_feats = batch['vis_feats'].cuda()
                boxes = batch['boxes'].cuda()

                input_ids = batch['word_ids'].cuda()
                token_type_ids = torch.zeros_like(input_ids)
                word_attention_mask = input_ids > 0

                ques_id = batch['question_ids']

                visn_feats = (vis_feats, boxes)
                sent_feats = (input_ids, token_type_ids, word_attention_mask)

                logit = self.model(visn_feats, sent_feats)
                score, predict = logit.max(1)

                predict = predict.cpu().numpy()
                # target = target.cpu().numpy()

                for qid, pred in zip(ques_id, predict):
                    pred_ans = loader.dataset.raw_dataset.label2ans[pred]
                    quesid2ans[qid] = pred_ans

            if dump_path is not None:
                loader.evaluator.dump_result(quesid2ans, dump_path)
            return quesid2ans

    def evaluate(self, loader, dump_path=None):
        evaluator = loader.evaluator
        quesid2ans = self.predict(loader, dump_path)
        return evaluator.evaluate(quesid2ans)

    @staticmethod
    def oracle_score(loader):
        evaluator = loader.evaluator
        quesid2ans = {}
        for i, batch in enumerate(loader):

            ques_id = batch['question_ids']
            label = batch['targets']

            _, label = label.max(1)
            for qid, l in zip(ques_id, label.cpu().numpy()):
                ans = loader.dataset.raw_dataset.label2ans[l]
                quesid2ans[qid] = ans
        return evaluator.evaluate(quesid2ans)

    def save(self, name):
        torch.save(self.model.state_dict(),
                   os.path.join(self.output, "%s.pth" % name))

    def load(self, path, loc=None):
        print("Load model from %s" % path)
        if loc is None:
            state_dict = torch.load("%s.pth" % path)
        else:
            state_dict = torch.load("%s.pth" % path, map_location=loc)
        self.model.load_state_dict(state_dict)


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

        num_answers = train_loader.dataset.raw_dataset.num_answers

        trainer = GQA(args, train_loader, val_loader, logger,
                      num_answers=num_answers, train=True)
        trainer.train()
    else:
        split = 'submit'

        print(f'Creating submission file on {split} split')
        train_loader = None
        transform = None
        test_loader = get_loader(
            args,
            split=split, mode='val', batch_size=args.batch_size,
            distributed=False, gpu=args.gpu,
            workers=args.num_workers, transform=transform,
            # topk=args.valid_topk, data_out=data_out)
        )
        num_answers = test_loader.dataset.raw_dataset.num_answers

        trainer = GQA(args, train_loader, test_loader, logger,
                      num_answers=num_answers, train=False)

        dump_path = 'Test_submission/gqa/' + args.comment + f'_{split}.json'
        print('dump_path is', dump_path)
        trainer.predict(test_loader, dump_path=dump_path)


if __name__ == "__main__":

    cudnn.benchmark = True
    # args = parse_args()
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
        comment += f'_imsize{args.resize_input_size}'
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
            'GQA_finetune', current_time + f'_GPU{args.world_size}_' + comment)
        args.log_dir = log_dir
        log_dir.mkdir(exist_ok=True, parents=True)
        print('logging at', log_dir)

    # if not args.dry:
    # from torch.utils.tensorboard import SummaryWriter
    #     writer = SummaryWriter(log_dir=log_dir)

    # nlvr2.train(nlvr2.train_tuple, nlvr2.valid_tuple)
    if args.distributed:
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(args,))
    else:
        main_worker(0, args)



# if __name__ == "__main__":
#     # Build Class
#     gqa = GQA()

#     # Load Model
#     if args.load is not None:
#         gqa.load(args.load)

#     # Test or Train
#     if args.test is not None:
#         args.fast = args.tiny = False       # Always loading all data in test
#         if 'submit' in args.test:
#             gqa.predict(
#                 get_tuple(args.test, bs=args.batch_size,
#                           shuffle=False, drop_last=False),
#                 dump=os.path.join(args.output, 'submit_predict.json')
#             )
#         if 'testdev' in args.test:
#             result = gqa.evaluate(
#                 get_tuple('testdev', bs=args.batch_size,
#                           shuffle=False, drop_last=False),
#                 dump=os.path.join(args.output, 'testdev_predict.json')
#             )
#             print(result)
#     else:
#         # print("Train Oracle: %0.2f" % (gqa.oracle_score(gqa.train_tuple) * 100))
#         print('Splits in Train data:', gqa.train_tuple.dataset.splits)
#         if gqa.valid_tuple is not None:
#             print('Splits in Valid data:', gqa.valid_tuple.dataset.splits)
#             print("Valid Oracle: %0.2f" % (gqa.oracle_score(gqa.valid_tuple) * 100))
#         else:
#             print("DO NOT USE VALIDATION")
#         gqa.train(gqa.train_tuple, gqa.valid_tuple)


