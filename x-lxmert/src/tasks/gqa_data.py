# coding=utf-8
# Copyleft 2019 project LXRT.

# import json

# import numpy as np
# import torch
# from torch.utils.data import Dataset

# from param import args
# from utils import load_obj_tsv


import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from multiprocessing import Pool
from pathlib import Path
import pickle
from tqdm import tqdm

from utils import box_position
from lxrt.tokenization import BertTokenizer
from param import args

import h5py



project_dir = Path(__file__).resolve().parent.parent.parent



# Load part of the dataset for fast checking.
# Notice that here is the number of images instead of the number of data,
# which means all related data to the images would be used.
# TINY_IMG_NUM = 512
# FAST_IMG_NUM = 5000


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
    def __init__(self, splits: str, verbose=True):
        self.name = splits
        self.splits = splits.split(',')

        # Loading datasets to data
        self.data = []
        for split in self.splits:
            self.data.extend(json.load(open("data/gqa/%s.json" % split)))
        if verbose:
            print("Load %d data from split(s) %s." % (len(self.data), self.name))

        # List to dict (for evaluation and others)
        self.id2datum = {
            datum['question_id']: datum
            for datum in self.data
        }

        # Answers
        self.ans2label = json.load(open("data/gqa/trainval_ans2label.json"))
        self.label2ans = json.load(open("data/gqa/trainval_label2ans.json"))
        assert len(self.ans2label) == len(self.label2ans)
        for ans, label in self.ans2label.items():
            assert self.label2ans[label] == ans

    @property
    def num_answers(self):
        return len(self.ans2label)

    def __len__(self):
        return len(self.data)


# class GQABufferLoader():
#     def __init__(self):
#         self.key2data = {}

#     def load_data(self, name, number):
#         if name == 'testdev':
#             path = "data/vg_gqa_imgfeat/gqa_testdev_obj36.tsv"
#         else:
#             path = "data/vg_gqa_imgfeat/vg_gqa_obj36.tsv"
#         key = "%s_%d" % (path, number)
#         if key not in self.key2data:
#             self.key2data[key] = load_obj_tsv(
#                 path,
#                 topk=number
#             )
#         return self.key2data[key]


# gqa_buffer_loader = GQABufferLoader()


# """
# Example in obj tsv:
# FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
#               "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]
# """
class GQATorchDataset(Dataset):
    def __init__(self, args, dataset: GQADataset, split, verbose, topk=-1):
        super().__init__()
        self.raw_dataset = dataset
        self.args = args
        self.verbose = verbose

        # Loading datasets to data
        self.sources = split.split(',')
        if self.verbose:
            print('Data sources: ', self.sources)

        self.img_ids_to_source = {}
        data_info_dicts = []

        for source in self.sources:
            data_info_path = project_dir.joinpath(f'data/gqa/{source}.json')
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

        if args.clustering:
            centroids_dir = Path(
                '/home/jaeminc/Dropbox/Projects/AI2/clustering/').resolve()
            if args.v4:
                centroids_dir = centroids_dir.joinpath('v4')
            imsize = args.resize_input_size
            if args.im_ratio == 'original':
                with open(centroids_dir.joinpath(f'{args.encoder}_{args.cluster_src}_gqa_img_id_to_cluster_id_{args.n_centroids}_iter{args.n_iter}_d{args.feat_dim}_grid{args.grid_size}.pkl'), 'rb') as f:
                    gqa_img_id_to_cluster_id = pickle.load(f)
            else:
                with open(centroids_dir.joinpath(f'{args.encoder}_{args.cluster_src}_gqa_img_id_to_cluster_id_{args.n_centroids}_iter{args.n_iter}_d{args.feat_dim}_grid{args.grid_size}_imsize{imsize}.pkl'), 'rb') as f:
                    gqa_img_id_to_cluster_id = pickle.load(f)

            # self.data_source_to_cluster_data = {
            #     'train': gqa_img_id_to_cluster_id,
            #     'testdev': gqa_img_id_to_cluster_id,
            #     'valid': gqa_img_id_to_cluster_id,
            #     'submit': gqa_img_id_to_cluster_id,
            # }
            self.img_id_to_cluster_id = gqa_img_id_to_cluster_id

        else:
            dataset_dir = Path(
                '/home/jaeminc/workspace/datasets').resolve()

            # if not len(list(dataset_dir.joinpath('GQA/features').iterdir())) < 1:
            #     dataset_dir = Path(
            #         '/net/nfs2.corp/prior/jaeminc/datasets').resolve()            

            if args.grid_model:
                if args.v4:
                    if args.im_ratio == 'original':
                        self.h5_path = dataset_dir.joinpath(
                            f'GQA/features/{args.encoder}_v4_grid{args.grid_size}.h5')

                        # self.data_source_to_h5_path = {
                        #     'train': dataset_dir.joinpath(f'GQA/features/{args.encoder}_v4_grid{args.grid_size}.h5'),
                        #     'testdev': dataset_dir.joinpath(f'GQA/features/{args.encoder}_v4_grid{args.grid_size}.h5'),
                        #     'valid': dataset_dir.joinpath(f'GQA/features/{args.encoder}_v4_grid{args.grid_size}.h5'),
                        #     'submit': dataset_dir.joinpath(f'GQA/features/{args.encoder}_v4_grid{args.grid_size}.h5'),
                        # }
                    else:
                        self.h5_path = dataset_dir.joinpath(
                            f'GQA/features/{args.encoder}_v4_grid{args.grid_size}_imsize{args.imsize}.h5')

                        # self.data_source_to_h5_path = {
                        #     'train': dataset_dir.joinpath(f'GQA/features/{args.encoder}_v4_grid{args.grid_size}_imsize{args.imsize}.h5'),
                        #     'testdev': dataset_dir.joinpath(f'GQA/features/{args.encoder}_v4_grid{args.grid_size}_imsize{args.imsize}.h5'),
                        #     'valid': dataset_dir.joinpath(f'GQA/features/{args.encoder}_v4_grid{args.grid_size}_imsize{args.imsize}.h5'),
                        #     'submit': dataset_dir.joinpath(f'GQA/features/{args.encoder}_v4_grid{args.grid_size}_imsize{args.imsize}.h5'),
                        # }
            else:
                self.h5_path = dataset_dir.joinpath(
                    f'GQA/features/maskrcnn_boxes36.h5')

                # self.data_source_to_h5_path = {
                #     'train': dataset_dir.joinpath(f'GQA/features/maskrcnn_boxes36.h5'),
                #     'testdev': dataset_dir.joinpath(f'GQA/features/maskrcnn_boxes36.h5'),
                #     'valid': dataset_dir.joinpath(f'GQA/features/maskrcnn_boxes36.h5'),
                #     'submit': dataset_dir.joinpath(f'GQA/features/maskrcnn_boxes36.h5'),
                # }

            assert self.h5_path.is_file(), self.h5_path

            # for source, path in self.data_source_to_h5_path.items():
            #     assert path.is_file(), (source, path)

            if self.verbose:
                print(f'Loading features from {self.h5_path}')

        self.h5_f = None

        self.data = data_info_dicts
        if self.verbose:
            # if 'sent' not in self.data_out:
                # print("# all images:", len(self.data))
            # else:
            print("# all sentences:", len(self.data))

        self.grid_size = args.grid_size
        self.n_grids = args.n_grids
        if self.args.grid_model:
            self.boxes = box_position(args.grid_size)
        else:
            self.n_boxes = args.n_boxes
            self.boxes = None

        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased",
            do_lower_case=True
        )

        self.max_text_length = args.max_text_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        datum = self.data[idx]
        img_id = datum['img_id']

        out_dict = {}

        ###### Image ######
        if self.args.clustering:
            # img_id_to_cluster_id = self.data_source_to_cluster_data[datum['source']]
            # cluster_id = img_id_to_cluster_id[img_id]
            cluster_id = self.img_id_to_cluster_id[img_id]
            assert cluster_id is not None, datum

            # cluster_id = datum['cluster_id']
            cluster_id = torch.from_numpy(cluster_id)
            out_dict['cluster_id'] = cluster_id
        else:
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

        if self.args.vis_sampling:
            sampled_idx = np.random.choice(
                self.n_grids, self.args.n_vis_sampling, replace=False)

            out_dict['boxes'] = boxes[sampled_idx]
            if self.args.clustering:
                out_dict['cluster_id'] = cluster_id[sampled_idx]
            else:
                out_dict['vis_feats'] = feats[sampled_idx]

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
        # # Zero-pad up to the sequence length.
        # while len(input_ids) < self.max_text_length:
        #     input_ids.append(0)
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


def collate_fn(batch, n_grids=25, max_text_length=20):
    B = len(batch)
    args = batch[0]['args']
    batch_out = {}

    B = len(batch)
    W_L = max(len(entry['word_ids']) for entry in batch)
    V_L = n_grids = len(batch[0]['boxes'])

    args = batch[0]['args']

    question_ids = []
    if args.clustering:
        cluster_ids = torch.zeros(B, V_L, dtype=torch.long)
    else:
        feat_dim = batch[0]['vis_feats'].size(1)
        vis_feats = torch.zeros(B, V_L, feat_dim, dtype=torch.float)
    boxes = torch.zeros(B, V_L, 4, dtype=torch.float)
    word_ids = torch.zeros(B, W_L, dtype=torch.long)
    # targets = torch.zeros(B, dtype=torch.long)
    targets = []
    for i, entry in enumerate(batch):
        question_ids.append(entry['question_id'])
        if args.clustering:
            cluster_ids[i] += entry['cluster_id']
        else:
            vis_feats[i] += entry['vis_feats']
        boxes[i] += entry['boxes']
        word_ids[i, :len(entry['word_ids'])] += entry['word_ids']
        if not args.test:
            targets.append(entry['target'])
    if not args.test:
        targets = torch.stack(targets, dim=0)

    batch_out = {}

    batch_out['question_ids'] = question_ids
    if args.clustering:
        batch_out['cluster_ids'] = cluster_ids
    else:
        batch_out['vis_feats'] = vis_feats
    batch_out['boxes'] = boxes
    batch_out['word_ids'] = word_ids
    if not args.test:
        batch_out['targets'] = targets

    return batch_out


def get_loader(args, split='train', mode='train',
               batch_size=32, workers=4, distributed=False, gpu=0,
               transform=None, data_out=['img']):

    verbose = (gpu == 0)

    _dset = GQADataset(split, verbose)

    if mode == 'train':
        topk = args.train_topk
    elif mode == 'val':
        topk = args.valid_topk
    else:
        topk = -1

    dataset = GQATorchDataset(args, _dset, split, verbose, topk)
    evaluator = GQAEvaluator(_dset)

    if distributed and mode == 'train':
        # sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank)
        train_sampler = DistributedSampler(dataset)
    else:
        train_sampler = None

    # if args.clustering:
    #     collate_func = collate_fn_cluster
    # else:
    collate_func = collate_fn

    if mode == 'train':
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=(train_sampler is None),
            num_workers=workers, pin_memory=True, sampler=train_sampler,
            collate_fn=collate_func)
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


