# coding=utf-8

import json
import numpy as np
from pathlib import Path
import pickle
from multiprocessing import Pool

from tqdm import tqdm
import h5py

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import LxmertTokenizer

from utils import box_position


class NLVR2Dataset:
    """
    An NLVR2 data example in json file:
    {
        "identifier": "train-10171-0-0",
        "img0": "train-10171-0-img0",
        "img1": "train-10171-0-img1",
        "label": 0,
        "sent": "An image shows one leather pencil case, displayed open with writing implements tucked inside.",
        "uid": "nlvr2_train_0"
    }
    """

    def __init__(self, args, splits: str, verbose=True):
        self.args = args
        self.datasets_dir = Path(self.args.datasets_dir)
        self.name = splits
        self.splits = splits.split(',')

        # Loading datasets to data
        self.data = []
        for split in self.splits:
            self.data.extend(json.load(open(self.datasets_dir.joinpath("data/nlvr2/%s.json" % split))))
        if verbose:
            print("Load %d data from split(s) %s." %
                  (len(self.data), self.name))

        # List to dict (for evaluation and others)
        self.id2datum = {}
        for datum in self.data:
            self.id2datum[datum['uid']] = datum

    def __len__(self):
        return len(self.data)


class NLVR2TorchDataset(Dataset):
    def __init__(self, args, dataset: NLVR2Dataset, split, verbose, topk=-1):
        super().__init__()
        self.raw_dataset = dataset
        self.args = args
        self.verbose = verbose

        self.datasets_dir = Path(self.args.datasets_dir)

        # Assign Clusters Ids
        data = self.raw_dataset.data

        if topk > 0:
            data = data[:topk]
            if self.verbose:
                print(f"Use only {topk} data")


        if args.grid_model:
            self.data_source_to_h5_path = {
                'train': self.datasets_dir.joinpath(f'nlvr2/features/{args.encoder}_train_v4_grid{args.grid_size}.h5'),
                'valid': self.datasets_dir.joinpath(f'nlvr2/features/{args.encoder}_valid_v4_grid{args.grid_size}.h5'),
                'test': self.datasets_dir.joinpath(f'nlvr2/features/{args.encoder}_test_v4_grid{args.grid_size}.h5'),
            }
        else:
            self.data_source_to_h5_path = {
                'train': self.datasets_dir.joinpath(f'nlvr2/features/maskrcnn_train_boxes36.h5'),
                'valid': self.datasets_dir.joinpath(f'nlvr2/features/maskrcnn_valid_boxes36.h5'),
                'test': self.datasets_dir.joinpath(f'nlvr2/features/maskrcnn_test_boxes36.h5'),
            }

        for source, path in self.data_source_to_h5_path.items():
            assert path.is_file(), (source, path)

        self.h5_path = self.data_source_to_h5_path[split]
        self.h5_f = None

        self.data = data

        if verbose:
            print("Use %d data in torch dataset" % (len(self.data)))
            print()

        self.grid_size = args.grid_size
        self.n_grids = self.grid_size ** 2
        if self.args.grid_model:
            self.boxes = box_position(args.grid_size)
        else:
            self.n_boxes = self.args.n_boxes
            self.boxes = None

        self.tokenizer = LxmertTokenizer.from_pretrained(
            "bert-base-uncased",
            do_lower_case=True
        )
        self.max_text_length = args.max_text_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        datum = self.data[idx]

        out_dict = {}

        ###### Image ######
        feats2 = []
        for key in ['img0', 'img1']:
            img_id = datum[key]
            f = self.h5_f
            if f is None:
                f = h5py.File(self.h5_path, 'r')
                self.h5_f = f

            if self.args.grid_model:
                feats = np.zeros(
                    shape=(self.grid_size, self.grid_size, self.args.feat_dim), dtype=np.float32)
                f[f'{img_id}/features'].read_direct(feats)
                feats = np.reshape(feats, (self.n_grids, self.args.feat_dim))
            else:
                feats = np.zeros(shape=(self.n_boxes, self.args.feat_dim), dtype=np.float32)
                f[f'{img_id}/features'].read_direct(feats)
            feats2.append(feats)
        feats = np.stack(feats2)  # [2, n_grids, feat_dim]
        feats = torch.from_numpy(feats)
        if self.args.grid_model:
            assert feats.size() == (2, self.n_grids, self.args.feat_dim), feats.size()
        else:
            assert feats.size() == (2, self.n_boxes, self.args.feat_dim), feats.size()

        if self.args.grid_model:
            boxes2 = [self.boxes, self.boxes]
        else:
            boxes2 = []
            for key in ['img0', 'img1']:
                img_id = datum[key]
                # Normalize the boxes (to 0 ~ 1)
                img_h = f[f'{img_id}/img_h'][()]
                img_w = f[f'{img_id}/img_w'][()]
                boxes = f[f'{img_id}/boxes'][()]
                boxes[:, (0, 2)] /= img_w
                boxes[:, (1, 3)] /= img_h
                # np.testing.assert_array_less(boxes, 1+1e-5)
                # np.testing.assert_array_less(-boxes, 0+1e-5)
                np.testing.assert_array_less(-boxes, 0+1e-5)

                boxes2.append(boxes)

        boxes = np.stack(boxes2)
        boxes = torch.from_numpy(boxes)
        boxes.clamp_(min=0.0, max=1.0)

        ###### Text #####
        question_id = datum['uid']
        question = datum['sent']
        tokens = self.tokenizer.tokenize(question.strip())

        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens) > self.max_text_length - 2:
            tokens = tokens[:(self.max_text_length - 2)]

        # concatenate lm labels and account for CLS, SEP, SEP
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = torch.LongTensor(input_ids)

        # Create target
        if 'label' in datum:
            label = datum['label']
        else:
            label = None

        out_dict['args'] = self.args

        out_dict['vis_feats'] = feats
        out_dict['boxes'] = boxes
        out_dict['input_ids'] = input_ids
        out_dict['label'] = label
        out_dict['question_id'] = question_id

        return out_dict

class NLVR2Evaluator:
    def __init__(self, dataset: NLVR2Dataset):
        self.dataset = dataset

    def evaluate(self, quesid2ans: dict):
        score = 0.
        for quesid, ans in quesid2ans.items():
            datum = self.dataset.id2datum[quesid]
            label = datum['label']
            if ans == label:
                score += 1
        return score / len(quesid2ans)

    def dump_result(self, quesid2ans: dict, path):
        """
        Dump result to a CSV file, which is compatible with NLVR2 evaluation system.
        NLVR2 CSV file requirement:
            Each line contains: identifier, answer

        :param quesid2ans: nlvr2 uid to ans (either "True" or "False")
        :param path: The desired path of saved file.
        :return:
        """
        with open(path, 'w') as f:
            for uid, ans in quesid2ans.items():
                idt = self.dataset.id2datum[uid]["identifier"]
                ans = 'True' if ans == 1 else 'False'
                f.write("%s,%s\n" % (idt, ans))


def collate_fn(batch):
    B = len(batch)
    W_L = max(len(entry['input_ids']) for entry in batch)
    V_L = batch[0]['boxes'].size(1)

    args = batch[0]['args']

    question_ids = []
    feat_dim = batch[0]['vis_feats'].size(2)
    vis_feats = torch.zeros(B, 2, V_L, feat_dim, dtype=torch.float)

    boxes = torch.zeros(B, 2, V_L, 4, dtype=torch.float)
    input_ids = torch.zeros(B, W_L, dtype=torch.long)
    if not args.test:
        labels = torch.zeros(B, dtype=torch.long)

    for i, entry in enumerate(batch):
        question_ids.append(entry['question_id'])

        vis_feats[i] += entry['vis_feats']
        boxes[i] += entry['boxes']
        input_ids[i, :len(entry['input_ids'])] += entry['input_ids']
        if not args.test:
            labels[i] += entry['label']

    batch_out = {}
    batch_out['question_ids'] = question_ids
    batch_out['vis_feats'] = vis_feats
    batch_out['boxes'] = boxes
    batch_out['word_ids'] = input_ids
    if not args.test:
        batch_out['labels'] = labels

    return batch_out


def get_loader(args, split='train', mode='train',
               batch_size=32, workers=4, distributed=False, gpu=0):

    verbose = (gpu == 0)

    _dset = NLVR2Dataset(split, verbose)

    if mode == 'train':
        topk = args.train_topk
    elif mode == 'val':
        topk = args.valid_topk
    else:
        topk = -1

    dataset = NLVR2TorchDataset(args, _dset, split, verbose, topk)
    evaluator = NLVR2Evaluator(_dset)

    if distributed and mode == 'train':
        train_sampler = DistributedSampler(dataset)
    else:
        train_sampler = None

    if mode == 'train':
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=(train_sampler is None),
            num_workers=workers, pin_memory=True, sampler=train_sampler,
            collate_fn=collate_fn)
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=True,
            sampler=None,
            collate_fn=collate_fn,
            drop_last=False)

    if verbose:
        loader.evaluator = evaluator

    return loader
