# coding=utf-8

import os
import collections
from pathlib import Path
import logging
import shutil

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from param import parse_args
from pretrain.qa_answer_table import load_lxmert_qa
from tasks.gqa_model import GQAModel
from tasks.gqa_data import get_loader
from utils import load_state_dict, LossMeter, count_parameters, reduce_dict, set_global_logging_level

set_global_logging_level(logging.ERROR, ["transformers"])
cudnn.benchmark = True


class Trainer:
    def __init__(self, args, train_loader=None, val_loader=None, logger=None, num_answers=0, train=True):
        self.args = args
        self.max_text_length = args.max_text_length
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_answers = num_answers
        self.logger = logger

        # Model
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
            path = args.load + '.pth'
            self.load(path, verbose=self.verbose)

        elif args.load_lxmert_qa is not None:
            path = args.load_lxmert_qa + '_LXRT.pth'
            load_lxmert_qa(args, path, self.model,
                           label2ans=self.train_loader.dataset.raw_dataset.label2ans,
                           verbose=self.verbose)

        # GPU Options
        print(f'Model Launching at GPU {self.args.gpu}')
        from time import time
        start = time()
        self.model.cuda(args.gpu)

        # Optimizer
        if train:
            self.optim, self.lr_scheduler = self.create_optimizer_and_scheduler()
            self.bce_loss = nn.BCEWithLogitsLoss()

        if args.multiGPU:
            assert args.distributed
            self.model = DDP(self.model, device_ids=[args.gpu],
                                find_unused_parameters=True
                                )

        if args.gpu == 0:
            print(f'It took {time() - start:.1f}s')

        # Output Directory
        self.output = args.output
        os.makedirs(self.output, exist_ok=True)

    def train(self):
        if self.verbose:
            loss_meter = LossMeter()
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

                vis_feats = batch['vis_feats'].cuda()
                boxes = batch['boxes'].cuda()

                input_ids = batch['word_ids'].cuda()
                target = batch['targets'].cuda()

                ques_id = batch['question_ids']

                B = len(batch['word_ids'])

                results = self.model(
                    input_ids=input_ids,
                    visual_feats=vis_feats,
                    visual_pos=boxes,
                    attention_mask=input_ids > 0,
                )
                logit = results['logit']

                assert logit.size() == target.size()
                assert logit.size() == (B, self.num_answers)

                loss = self.bce_loss(logit, target)

                loss.backward()

                if update:
                    if not self.args.no_clip_grad:
                        nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.args.clip_grad_norm)

                    self.optim.step()
                    self.lr_scheduler.step()
                    for param in self.model.parameters():
                        param.grad = None

                try:
                    lr = self.scheduler.get_last_lr()[0]
                except AttributeError:
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

                vis_feats = batch['vis_feats'].cuda()
                boxes = batch['boxes'].cuda()

                input_ids = batch['word_ids'].cuda()
                ques_id = batch['question_ids']

                results = self.model(
                    input_ids=input_ids,
                    visual_feats=vis_feats,
                    visual_pos=boxes,
                    attention_mask=input_ids > 0,
                )
                logit = results['logit']
                score, predict = logit.max(1)

                predict = predict.cpu().numpy()

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

    def load(self, path, loc='cpu', verbose=False):
        state_dict = load_state_dict(path, loc)
        results = self.model.load_state_dict(state_dict, strict=False)
        if verbose:
            print('Loaded from ', path)
            print(results)
        # self.start_epoch = int(self.args.load.split('Epoch')[-1])


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
        train_loader = get_loader(
            args,
            # 'mscoco_minival', mode='train', batch_size=args.batch_size,
            split=args.train, mode='train', batch_size=args.batch_size,
            distributed=args.distributed, gpu=args.gpu,
            workers=args.num_workers,
        )

        val_loader = get_loader(
            args,
            split=args.valid, mode='val', batch_size=args.batch_size,
            distributed=args.distributed, gpu=args.gpu,
            workers=4,
        )

        num_answers = train_loader.dataset.raw_dataset.num_answers

        trainer = Trainer(args, train_loader, val_loader, logger,
                      num_answers=num_answers, train=True)
        trainer.train()
    else:
        split = 'submit'

        print(f'Creating submission file on {split} split')
        train_loader = None
        test_loader = get_loader(
            args,
            split=split, mode='val', batch_size=args.batch_size,
            distributed=False, gpu=args.gpu,
            workers=args.num_workers,
        )
        num_answers = test_loader.dataset.raw_dataset.num_answers

        trainer = Trainer(args, train_loader, test_loader, logger,
                      num_answers=num_answers, train=False)

        dump_path = 'Test_submission/gqa/' + args.comment + f'_{split}.json'
        print('dump_path is', dump_path)
        trainer.predict(test_loader, dump_path=dump_path)


if __name__ == "__main__":
    args = parse_args()
    print(args)
    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node
    args.n_grids = args.grid_size ** 2

    if args.grid_model:
        comment = f'Grid{args.grid_size}'
    else:
        comment = f'Box{args.n_boxes}'
    comment += f'_{args.backbone}'
    comment += f'_{args.encoder}'

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

    if args.distributed:
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(args,))
    else:
        main_worker(0, args)
