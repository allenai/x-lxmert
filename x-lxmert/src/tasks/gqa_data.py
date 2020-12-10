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


class GQADataset:
    """
    A GQA data example in json file:
    {
        "img_id": "2375429",
        "label": {
            "pipe": 1.0
        },
        "question_id": "07333408",
        "sent": "What is on the white wall?"
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
            self.data.extend(json.load(open(self.datasets_dir.joinpath("ddata/gqa/%s.json" % split))))
        if verbose:
            print("Load %d data from split(s) %s." % (len(self.data), self.name))

        # List to dict (for evaluation and others)
        self.id2datum = {
            datum['question_id']: datum
            for datum in self.data
        }

        # Answers
        self.ans2label = json.load(open(self.datasets_dir.joinpath("ddata/gqa/trainval_ans2label.json")))
        self.label2ans = json.load(open(self.datasets_dir.joinpath("ddata/gqa/trainval_label2ans.json")))
        assert len(self.ans2label) == len(self.label2ans)
        for ans, label in self.ans2label.items():
            assert self.label2ans[label] == ans

    @property
    def num_answers(self):
        return len(self.ans2label)

    def __len__(self):
        return len(self.data)


class GQATorchDataset(Dataset):
    def __init__(self, args, dataset: GQADataset, split, verbose, topk=-1):
        super().__init__()
        self.raw_dataset = dataset
        self.args = args
        self.verbose = verbose

        self.datasets_dir = Path(self.args.datasets_dir)

        # Loading datasets to data
        self.sources = split.split(',')
        if self.verbose:
            print('Data sources: ', self.sources)

        self.img_ids_to_source = {}
        data_info_dicts = []

        for source in self.sources:
            data_info_path = self.datasets_dir.joinpath(f'data/gqa/{source}.json')
            with open(data_info_path) as f:
                _data_info_dicts = json.load(f)
                # source_img_ids.append([d['img_id'] for d in _data_info_dicts])
                for _d in _data_info_dicts:
                    self.img_ids_to_source[_d['img_id']] = source
                    _d['source'] = source
                data_info_dicts.extend(_data_info_dicts)
            if self.verbose:
                print(f"Loaded {len(_data_info_dicts)} data from", source)

        if topk > 0:
            data_info_dicts = data_info_dicts[:topk]
            if self.verbose:
                print(f"Use only {topk} data")


        if args.grid_model:
            self.h5_path = self.datasets_dir.joinpath(f'GQA/features/{args.encoder}_grid{args.grid_size}.h5')

        else:
            self.h5_path = self.datasets_dir.joinpath(f'GQA/features/maskrcnn_boxes36.h5')
        assert self.h5_path.is_file(), self.h5_path

        if self.verbose:
            print(f'Loading features from {self.h5_path}')

        self.h5_f = None

        self.data = data_info_dicts
        if self.verbose:
            print("# all sentences:", len(self.data))

        self.grid_size = args.grid_size
        self.n_grids = args.n_grids
        if self.args.grid_model:
            self.boxes = box_position(args.grid_size)
        else:
            self.n_boxes = args.n_boxes
            self.boxes = None

        self.tokenizer = LxmertTokenizer.from_pretrained('unc-nlp/lxmert-base-uncased')

        self.max_text_length = args.max_text_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        datum = self.data[idx]
        img_id = datum['img_id']

        out_dict = {}

        ###### Image ######
        f = self.h5_f
        if f is None:
            f = h5py.File(self.h5_path, 'r')
            self.h5_f = f

        if self.args.grid_model:
            feats = np.zeros(
                shape=(self.grid_size, self.grid_size, self.args.feat_dim), dtype=np.float32)
            f[f'{img_id}/features'].read_direct(feats)
            feats = np.reshape(feats, (self.n_grids, self.args.feat_dim))
            feats = torch.from_numpy(feats)
        else:
            feats = np.zeros(
                shape=(self.n_boxes, self.args.feat_dim), dtype=np.float32)
            f[f'{img_id}/features'].read_direct(feats)
            feats = torch.from_numpy(feats)
        out_dict['vis_feats'] = feats

        if self.args.grid_model:
            boxes = self.boxes
            boxes = torch.from_numpy(boxes)
        else:
            # Normalize the boxes (to 0 ~ 1)
            img_h = f[f'{img_id}/img_h'][()]
            img_w = f[f'{img_id}/img_w'][()]
            boxes = f[f'{img_id}/boxes'][()]
            boxes[:, (0, 2)] /= img_w
            boxes[:, (1, 3)] /= img_h
            # np.testing.assert_array_less(boxes, 1+1e-5)
            # np.testing.assert_array_less(boxes, 1+5e-2)
            # np.testing.assert_array_less(-boxes, 0+1e-5)

            np.testing.assert_array_less(-boxes, 0+1e-5)
            boxes = torch.from_numpy(boxes)

            boxes.clamp_(min=0.0, max=1.0)

        out_dict['boxes'] = boxes

        out_dict['question_id'] = datum['question_id']
        question = datum['sent']
        out_dict['question'] = question

        tokens = self.tokenizer.tokenize(question.strip())
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens) > self.max_text_length - 2:
            tokens = tokens[:(self.max_text_length - 2)]
        # concatenate lm labels and account for CLS, SEP, SEP
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = torch.LongTensor(input_ids)
        out_dict['word_ids'] = input_ids

        # Provide label (target)
        if 'label' in datum:
            label = datum['label']
            target = torch.zeros(self.raw_dataset.num_answers)
            for ans, score in label.items():
                if ans in self.raw_dataset.ans2label:
                    target[self.raw_dataset.ans2label[ans]] = score
        else:
            target = None

        out_dict['target'] = target

        out_dict['args'] = self.args

        return out_dict


def collate_fn(batch):
    B = len(batch)
    args = batch[0]['args']
    batch_out = {}

    B = len(batch)
    W_L = max(len(entry['word_ids']) for entry in batch)
    V_L = len(batch[0]['boxes'])

    args = batch[0]['args']

    question_ids = []
    feat_dim = batch[0]['vis_feats'].size(1)
    vis_feats = torch.zeros(B, V_L, feat_dim, dtype=torch.float)
    boxes = torch.zeros(B, V_L, 4, dtype=torch.float)
    word_ids = torch.zeros(B, W_L, dtype=torch.long)
    # targets = torch.zeros(B, dtype=torch.long)
    targets = []
    for i, entry in enumerate(batch):
        question_ids.append(entry['question_id'])
        vis_feats[i] += entry['vis_feats']
        boxes[i] += entry['boxes']
        word_ids[i, :len(entry['word_ids'])] += entry['word_ids']
        if not args.test:
            targets.append(entry['target'])
    if not args.test:
        targets = torch.stack(targets, dim=0)

    batch_out = {}

    batch_out['question_ids'] = question_ids
    batch_out['vis_feats'] = vis_feats
    batch_out['boxes'] = boxes
    batch_out['word_ids'] = word_ids
    if not args.test:
        batch_out['targets'] = targets

    return batch_out


class GQAEvaluator:
    def __init__(self, dataset: GQADataset):
        self.dataset = dataset

    def evaluate(self, quesid2ans: dict):
        score = 0.
        for quesid, ans in quesid2ans.items():
            datum = self.dataset.id2datum[quesid]
            label = datum['label']
            if ans in label:
                score += label[ans]
        return score / len(quesid2ans)

    def dump_result(self, quesid2ans: dict, path):
        """
        Dump the result to a GQA-challenge submittable json file.
        GQA json file submission requirement:
            results = [result]
            result = {
                "questionId": str,      # Note: it's a actually an int number but the server requires an str.
                "prediction": str
            }

        :param quesid2ans: A dict mapping question id to its predicted answer.
        :param path: The file path to save the json file.
        :return:
        """
        with open(path, 'w') as f:
            result = []
            for ques_id, ans in quesid2ans.items():
                result.append({
                    'questionId': ques_id,
                    'prediction': ans
                })
            json.dump(result, f, indent=4, sort_keys=True)


def get_loader(args, split='train', mode='train',
               batch_size=32, workers=4, distributed=False, gpu=0):

    verbose = (gpu == 0)

    _dset = GQADataset(args, split, verbose)

    if mode == 'train':
        topk = args.train_topk
    elif mode == 'val':
        topk = args.valid_topk
    else:
        topk = -1

    dataset = GQATorchDataset(args, _dset, split, verbose, topk)
    evaluator = GQAEvaluator(_dset)

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
            collate_fn=collate_func,
            drop_last=False)

    if verbose:
        loader.evaluator = evaluator

    return loader
