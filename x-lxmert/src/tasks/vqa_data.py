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


class VQADataset:
    """
    A VQA data example in json file:
        {
            "answer_type": "other",
            "img_id": "COCO_train2014_000000458752",
            "label": {
                "net": 1
            },
            "question_id": 458752000,
            "question_type": "what is this",
            "sent": "What is this photo taken looking through?"
        }
    """
    def __init__(self, args, splits: str, verbose=True):
        self.args = args
        self.datasets_dir = Path(self.args.datasets_dir)
        self.name = splits
        self.splits = splits.split(',')

        # Loading datasets
        self.data = []
        for split in self.splits:
            self.data.extend(json.load(open(self.datasets_dir.joinpath("data/vqa/%s.json" % split))))
        if verbose:
            print("Load %d data from split(s) %s." % (len(self.data), self.name))

        # Convert list to dict (for evaluation)
        self.id2datum = {
            datum['question_id']: datum
            for datum in self.data
        }

        # Answers
        self.ans2label = json.load(open(self.datasets_dir.joinpath("data/vqa/trainval_ans2label.json")))
        self.label2ans = json.load(open(self.datasets_dir.joinpath("data/vqa/trainval_label2ans.json")))
        assert len(self.ans2label) == len(self.label2ans)

        if verbose:
            print('# Answers:', len(self.ans2label))

    @property
    def num_answers(self):
        return len(self.ans2label)

    def __len__(self):
        return len(self.data)


class VQATorchDataset(Dataset):
    def __init__(self, args, dataset: VQADataset, split, verbose, topk=-1):
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
            data_info_path = self.datasets_dir.joinpath(f'data/vqa/{source}.json')
            with open(data_info_path) as f:
                _data_info_dicts = json.load(f)
                # source_img_ids.append([d['img_id'] for d in _data_info_dicts])
                for _d in _data_info_dicts:
                    self.img_ids_to_source[_d['img_id']] = source
                    _d['source'] = source
                data_info_dicts.extend(_data_info_dicts)
            if self.verbose:
                print(f"Loaded {len(_data_info_dicts)} data from", source)

        # data_info_dicts = self.raw_dataset.data

        if topk > 0:
            data_info_dicts = data_info_dicts[:topk]
            if self.verbose:
                print(f"Use only {topk} data")

        if args.grid_model:
            self.data_source_to_h5_path = {
                'train': self.datasets_dir.joinpath(f'COCO/features/{args.encoder}_train_grid{args.grid_size}.h5'),
                'minival': self.datasets_dir.joinpath(f'COCO/features/{args.encoder}_valid_grid{args.grid_size}.h5'),
                'nominival': self.datasets_dir.joinpath(f'COCO/features/{args.encoder}_valid_grid{args.grid_size}.h5'),
                'test': self.datasets_dir.joinpath(f'COCO/features/{args.encoder}_test_grid{args.grid_size}.h5'),
            }
        else:
            self.data_source_to_h5_path = {
                'train': self.datasets_dir.joinpath(f'COCO/features/maskrcnn_train_boxes36.h5'),
                'minival': self.datasets_dir.joinpath(f'COCO/features/maskrcnn_valid_boxes36.h5'),
                'nominival': self.datasets_dir.joinpath(f'COCO/features/maskrcnn_valid_boxes36.h5'),
                'test': self.datasets_dir.joinpath(f'COCO/features/maskrcnn_test_boxes36.h5'),
            }

        for source, path in self.data_source_to_h5_path.items():
            assert path.is_file(), (source, path)

        self.source_to_h5 = None

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
        if self.source_to_h5 is None:
            self.source_to_h5 = {}
            for source, path in self.data_source_to_h5_path.items():
                self.source_to_h5[source] = None
        source = self.img_ids_to_source[img_id]
        f = self.source_to_h5[source]
        if f is None:
            path = self.data_source_to_h5_path[source]
            f = h5py.File(path, 'r')
            self.source_to_h5[source] = f

        if self.args.grid_model:
            feats = np.zeros(
                shape=(self.grid_size, self.grid_size, self.args.feat_dim), dtype=np.float32)
            f[f'{img_id}/features'].read_direct(feats)
            feats = np.reshape(feats, (self.n_grids, self.args.feat_dim))
            feats = torch.from_numpy(feats)
        else:
            feats = np.zeros(shape=(self.n_boxes, self.args.feat_dim), dtype=np.float32)
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


class VQAEvaluator:
    def __init__(self, dataset: VQADataset):
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
        Dump results to a json file, which could be submitted to the VQA online evaluation.
        VQA json file submission requirement:
            results = [result]
            result = {
                "question_id": int,
                "answer": str
            }

        :param quesid2ans: dict of quesid --> ans
        :param path: The desired path of saved file.
        """
        with open(path, 'w') as f:
            result = []
            for ques_id, ans in quesid2ans.items():
                result.append({
                    'question_id': ques_id,
                    'answer': ans
                })
            json.dump(result, f, indent=4, sort_keys=True)


def get_loader(args, split='train', mode='train',
               batch_size=32, workers=4, distributed=False, gpu=0):


    verbose = (gpu == 0)

    _dset = VQADataset(args, split, verbose)

    if mode == 'train':
        topk = args.train_topk
    elif mode == 'val':
        topk = args.valid_topk
    else:
        topk = -1

    dataset = VQATorchDataset(args, _dset, split, verbose, topk)
    evaluator = VQAEvaluator(_dset)

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
