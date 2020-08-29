# coding=utf-8
# Copyleft 2019 project LXRT.


from pathlib import Path
from collections import defaultdict
import json
import random
from multiprocessing import Pool
import h5py
import pickle
import math
from tqdm import tqdm
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from torchvision import transforms
from torchvision.datasets.folder import default_loader

from lxrt.tokenization import BertTokenizer
from pretrain.qa_answer_table import AnswerTable
# from utils import load_feature_split

tokenizer = BertTokenizer.from_pretrained(
    "bert-base-uncased",
    do_lower_case=True
)

def text_process(sent, max_text_length=20, PAD_ID=0):
    tokens = tokenizer.tokenize(sent.strip())

    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens) > max_text_length - 2:
        tokens = tokens[:(max_text_length - 2)]
    tokens = ['[CLS]'] + tokens + ['[SEP]']
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    n_tokens = len(input_ids)

    # # Pad up to the sequence length.
    # while len(input_ids) < max_text_length:
    #     input_ids.append(PAD_ID)

    return input_ids, n_tokens


project_dir = Path(__file__).resolve().parent.parent.parent
coco_img_dir = Path('/home/jaeminc/workspace/datasets/COCO/images/').resolve()


def get_datum(datum):
    data = []
    _sents = []

    # sent_sources = datum['sent_sources']
    caption_only = datum['caption_only']
    qa = datum['qa']

    for text_source, sents in datum['sentf'].items():
        if caption_only:
            if text_source not in ['mscoco', 'vg']:
                continue

        if qa:
            if text_source in datum['labelf']:
                labels = datum['labelf'][text_source]
            else:
                labels = None

        img_id = datum['img_id']

        for sent_idx, sent in enumerate(sents):
            # remove duplicate sentence
            if sent not in _sents:
                input_ids, n_tokens = text_process(sent, max_text_length=20)

                new_datum = {
                    # 'uid': make_uid(img_id, dataset, sent_id),
                    'img_id': img_id,
                    # 'img_path': img_path,
                    # 'img': img,
                    'sent': sent,
                    'input_ids': input_ids,
                    'n_tokens': n_tokens,
                    # 'cluster_id': None
                    'img_source': datum['img_source'],
                    'text_source': text_source
                }

                new_datum['uid'] = make_uid(img_id, text_source, sent_idx)

                if qa:
                    if labels is not None:
                        label = labels[sent_idx]
                        # assert len(label) > 0, (img_id, labels, sent_idx, label)
                        # can have len = 0
                        new_datum['label'] = label

                data.append(new_datum)
            _sents.append(sent)
    return data


def add_img_id(datum):
    return datum['img_id']


class COCODataset(Dataset):
    def __init__(self, split='mscoco_mininval', coco_cap_only=True, image_only=True,
                 loader=default_loader, transform=None, topk=-1, data_out=['img'], verbose=True, args=None):

        self.loader = loader
        self.transform = transform
        self.data_out = data_out
        self.topk = topk
        self.verbose = verbose
        self.args = args

        if 'img' in self.data_out:
            if split == 'mscoco_train':
                self.coco_img_dir = coco_img_dir.joinpath('train2014')
            elif split == 'mscoco_minival':
                self.coco_img_dir = coco_img_dir.joinpath('val2014')
            elif split == 'mscoco_nominival':
                self.coco_img_dir = coco_img_dir.joinpath('val2014')
            if self.verbose:
                print(split)
                print('# images:', len(list(self.coco_img_dir.iterdir())))

        # Loading datasets to data
        self.sources = split.split(',')
        if self.verbose:
            print('Data sources: ', self.sources)

        # source_img_ids = []

        self.answer_table = AnswerTable()
        # if self.verbose:
        print("Load an answer table of size %d." % (len(self.answer_table.ans2id_map())))

        self.img_ids_to_source = {}

        data = []
        for img_source in self.sources:
            data_info_path = project_dir.joinpath(f'data/lxmert/{img_source}.json')
            with open(data_info_path) as f:
                _data = json.load(f)
                if self.verbose:
                    print(f"Loaded {len(_data)} data from", img_source)
                # source_img_ids.append([d['img_id'] for d in _data])
                for datum in _data:
                    self.img_ids_to_source[datum['img_id']] = img_source
                    datum['img_source'] = img_source
                    datum['caption_only'] = args.caption_only
                    datum['clustering'] = args.clustering
                    datum['max_text_length'] = args.max_text_length
                    datum['qa'] = args.task_qa

                data.extend(_data)

        # Modify the answers
        if args.task_qa:
            for datum in data:
                labelf = datum['labelf']
                for _qa_source, labels in labelf.items():
                    for label in labels:
                        for ans in list(label.keys()):
                            new_ans = self.answer_table.convert_ans(ans)
                            if self.answer_table.used(new_ans):
                                if ans != new_ans:
                                    label[new_ans] = label.pop(ans)
                            else:
                                label.pop(ans)

        if self.topk > 0:
            data = data[:self.topk]
            if self.verbose:
                print(f"Use only {self.topk} data")

        if args.task_qa:
            self.evaluator = QAEvaluator(data)


        if args.clustering or args.target_cluster:
            centroids_dir = Path('../../../datasets/cluster_centroids').resolve()
            imsize = args.resize_input_size
            if args.im_ratio == 'original':
                with open(centroids_dir.joinpath(f'{args.encoder}_{args.cluster_src}_mscoco_train_img_id_to_cluster_id_{args.n_centroids}_iter{args.n_iter}_d{args.feat_dim}_grid{args.grid_size}.pkl'), 'rb') as f:
                    mscoco_train_img_id_to_cluster_id = pickle.load(f)
                with open(centroids_dir.joinpath(f'{args.encoder}_{args.cluster_src}_mscoco_valid_img_id_to_cluster_id_{args.n_centroids}_iter{args.n_iter}_d{args.feat_dim}_grid{args.grid_size}.pkl'), 'rb') as f:
                    mscoco_valid_img_id_to_cluster_id = pickle.load(f)
                with open(centroids_dir.joinpath(f'{args.encoder}_{args.cluster_src}_vg_img_id_to_cluster_id_{args.n_centroids}_iter{args.n_iter}_d{args.feat_dim}_grid{args.grid_size}.pkl'), 'rb') as f:
                    vg_img_id_to_cluster_id = pickle.load(f)
            else:
                with open(centroids_dir.joinpath(f'{args.encoder}_{args.cluster_src}_mscoco_train_img_id_to_cluster_id_{args.n_centroids}_iter{args.n_iter}_d{args.feat_dim}_grid{args.grid_size}_imsize{imsize}.pkl'), 'rb') as f:
                    mscoco_train_img_id_to_cluster_id = pickle.load(f)
                with open(centroids_dir.joinpath(f'{args.encoder}_{args.cluster_src}_mscoco_valid_img_id_to_cluster_id_{args.n_centroids}_iter{args.n_iter}_d{args.feat_dim}_grid{args.grid_size}_imsize{imsize}.pkl'), 'rb') as f:
                    mscoco_valid_img_id_to_cluster_id = pickle.load(f)
                with open(centroids_dir.joinpath(f'{args.encoder}_{args.cluster_src}_vg_img_id_to_cluster_id_{args.n_centroids}_iter{args.n_iter}_d{args.feat_dim}_grid{args.grid_size}_imsize{imsize}.pkl'), 'rb') as f:
                    vg_img_id_to_cluster_id = pickle.load(f)

            self.data_source_to_cluster_data = {
                'mscoco_train': mscoco_train_img_id_to_cluster_id,
                'mscoco_minival': mscoco_valid_img_id_to_cluster_id,
                'mscoco_nominival': mscoco_valid_img_id_to_cluster_id,
                'vgnococo': vg_img_id_to_cluster_id
            }

        with Pool(8) as pool:
            if self.verbose:
                # for _data in tqdm(pool.imap(get_datum, data),
                #                   total=len(data), ncols=100):
                #     new_data.extend(_data)
                data = [datum for _data in tqdm(pool.imap(get_datum, data), total=len(data), ncols=100) for datum in _data]
            else:
                # for _data in pool.imap(get_datum, data):
                #     new_data.extend(_data)
                data = [datum for _data in pool.imap(get_datum, data) for datum in _data]

        if self.args.target_exact_feat or self.args.target_prob or self.args.feed_exact_feat or self.args.target_obj_id:
            centroids_dir = Path('../../../datasets/').resolve()
            if args.grid_model:
                if args.im_ratio == 'original':
                    self.data_source_to_h5_path = {
                        'mscoco_train': dataset_dir.joinpath(f'COCO/features/{args.encoder}_train_grid{args.grid_size}.h5'),
                        'mscoco_minival': dataset_dir.joinpath(f'COCO/features/{args.encoder}_valid_grid{args.grid_size}.h5'),
                        'mscoco_nominival': dataset_dir.joinpath(f'COCO/features/{args.encoder}_valid_grid{args.grid_size}.h5'),
                        'vgnococo': dataset_dir.joinpath(f'VG/features/{args.encoder}_grid{args.grid_size}.h5'),
                    }
                else:
                    self.data_source_to_h5_path = {
                        'msoco_train': dataset_dir.joinpath(f'COCO/features/{args.encoder}_train_grid{args.grid_size}_imsize{args.imsize}.h5'),
                        'msoco_minival': dataset_dir.joinpath(f'COCO/features/{args.encoder}_valid_grid{args.grid_size}_imsize{args.imsize}.h5'),
                        'msoco_nominival': dataset_dir.joinpath(f'COCO/features/{args.encoder}_valid_grid{args.grid_size}_imsize{args.imsize}.h5'),
                        'vgnococo': dataset_dir.joinpath(f'VG/features/{args.encoder}_grid{args.grid_size}_imsize{args.imsize}.h5'),
                    }

            else:
                self.data_source_to_h5_path = {
                    'mscoco_train': dataset_dir.joinpath(f'COCO/features/maskrcnn_train_boxes36.h5'),
                    'mscoco_minival': dataset_dir.joinpath(f'COCO/features/maskrcnn_valid_boxes36.h5'),
                    'mscoco_nominival': dataset_dir.joinpath(f'COCO/features/maskrcnn_valid_boxes36.h5'),
                    'vgnococo': dataset_dir.joinpath(f'VG/features/maskrcnn_boxes36.h5'),
                }

            for source, path in self.data_source_to_h5_path.items():
                assert path.is_file(), (source, path)

            self.source_to_h5 = None

        self.data = data

        if args.vis_mask_COCO_only:
            COCO_data = []
            for datum in self.data:
                if datum['text_source'] == 'mscoco' and 'mscoco' in datum['img_source']:
                    COCO_data.append(datum)
            self.COCO_data = COCO_data
            if self.verbose:
                print('# COCO captions:', len(self.COCO_data))
        elif args.vis_mask_COCOVG_only:
            COCO_data = []
            for datum in self.data:
                if datum['text_source'] in ['mscoco', 'vg'] and 'mscoco' in datum['img_source']:
                    COCO_data.append(datum)
            self.COCO_data = COCO_data
            if self.verbose:
                print('# COCO captions:', len(self.COCO_data))

        if self.verbose:
            if 'sent' not in self.data_out:
                print("# all images:", len(self.data))
            else:
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

        ###### Pretrainining Objective ######
        tasks = []
        if self.args.task_mask_lm:
            tasks.append('word_mask')
        if self.args.task_obj_predict:
            tasks.append('vis_mask')
        if self.args.task_matched:
            tasks.append('matched')
        if self.args.task_qa:
            tasks.append('qa')
        self.tasks = tasks

        if self.verbose:
            print('data_out:', self.data_out)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        out_dict = {}

        datum = self.data[idx]
        uid = datum['uid']
        out_dict['uid'] = uid

        ###### Image ######
        img_id = datum['img_id']
        if 'cluster_id' in self.data_out:
            # cluster_id = datum['cluster_id']
            img_id_to_cluster_id = self.data_source_to_cluster_data[datum['img_source']]
            cluster_id = img_id_to_cluster_id[img_id]
            assert cluster_id is not None, datum

            cluster_id = torch.from_numpy(cluster_id)
            out_dict['cluster_id'] = cluster_id

        if self.args.target_exact_feat or self.args.target_prob or self.args.feed_exact_feat or self.args.target_obj_id:
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

            if 'feat' in self.data_out:
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

            if self.args.target_prob:
                # ResNet obj probs
                if 'prob' in self.data_out:
                    probs = np.zeros(
                        shape=(self.grid_size, self.grid_size, self.args.n_class), dtype=np.float32)
                    f[f'{img_id}/probs'].read_direct(probs)
                    probs = np.reshape(probs, (self.n_grids, self.args.n_class))
                    probs = torch.from_numpy(probs)
                    out_dict['cls_probs'] = probs

                # Faster-RCNN obj probs
                if 'obj_prob' in self.data_out:
                    obj_probs = np.zeros(shape=(self.n_boxes, 1600), dtype=np.float32)
                    f[f'{img_id}/probs'].read_direct(obj_probs)
                    obj_probs = torch.from_numpy(obj_probs)
                    out_dict['obj_probs'] = obj_probs

                # Faster-RCNN obj probs
                if 'attr_prob' in self.data_out:
                    attr_probs = np.zeros(shape=(self.n_boxes, 400), dtype=np.float32)
                    f[f'{img_id}/probs'].read_direct(attr_probs)
                    attr_probs = torch.from_numpy(attr_probs)
                    out_dict['attr_probs'] = attr_probs

            if 'obj_id' in self.data_out:
                obj_id = np.zeros(shape=(self.n_boxes), dtype=int)
                f[f'{img_id}/obj_id'].read_direct(obj_id)
                obj_id = torch.from_numpy(obj_id)
                out_dict['obj_id'] = obj_id

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
            np.testing.assert_array_less(-boxes, 0+1e-5)
            boxes = torch.from_numpy(boxes)

            boxes.clamp_(min=0.0, max=1.0)

        out_dict['boxes'] = boxes

        if self.args.vis_sampling:
            sampled_idx = np.random.choice(self.n_grids, self.args.n_vis_sampling, replace=False)

            out_dict['boxes'] =  boxes[sampled_idx]
            if 'cluster_id' in self.data_out:
                out_dict['cluster_id'] = cluster_id[sampled_idx]
            if 'feat' in self.data_out:
                out_dict['vis_feats'] = feats[sampled_idx]

        ###### Text #####
        sent = datum['sent']
        # input_ids, n_tokens = text_process(sent)
        input_ids, n_tokens = datum['input_ids'], datum['n_tokens']
        input_ids = torch.LongTensor(input_ids)

        out_dict['sent'] = sent
        out_dict['input_ids'] = input_ids
        out_dict['n_tokens'] = n_tokens

        # Flip -> Img-Text not matched
        if 'matched' in self.data_out and random.random() < 0.5:
            other_datum = self.data[random.randint(0, len(self.data) - 1)]
            while img_id == other_datum['img_id']:
                other_datum = self.data[random.randint(0, len(self.data) - 1)]
            other_sent = other_datum['sent']
            # other_input_ids, other_n_tokens = text_process(other_sent)
            other_input_ids, other_n_tokens = other_datum['input_ids'], other_datum['n_tokens']

            other_input_ids = torch.LongTensor(other_input_ids)

            out_dict['matched_label'] = 0
            out_dict['other_sent'] = other_sent
            out_dict['other_input_ids'] = other_input_ids
            out_dict['other_n_tokens'] = other_n_tokens
        else:
            out_dict['matched_label'] = 1
            # out_dict['other_sent'] = sent
            # out_dict['other_input_ids'] = input_ids
            out_dict['other_n_tokens'] = n_tokens

        if 'img' in self.data_out:
            img_id = datum['img_id']
            if self.coco_img_dir.joinpath(img_id + '.jpg').is_file():
                img_path = self.coco_img_dir.joinpath(img_id + '.jpg')
            elif self.coco_img_dir.joinpath(img_id + '.png').is_file():
                img_path = self.coco_img_dir.joinpath(img_id + '.png')
            assert img_path is not None
            img = self.loader(img_path)
            img = self.transform(img)

            out_dict['img'] = img

        if 'word_mask_idx' in self.data_out:
            total_idx = list(range(1, 1 + n_tokens))  # Don't mask CLS / SEP
            n_masks = random.randint(1, n_tokens)
            word_mask_idx = np.random.choice(total_idx, n_masks, replace=False)
            word_mask_idx = torch.from_numpy(word_mask_idx)

            out_dict['word_mask_idx'] = word_mask_idx

        if self.args.task_qa:
            # Label, convert answer to id
            if 'label' in datum:
                label = datum['label'].copy()
                if len(label) > 0:

                    for ans in list(label.keys()):
                        label[self.answer_table.ans2id(ans)] = label.pop(ans)
                    keys, values = zip(*label.items())
                    # single answer
                    if len(keys) == 1:
                        ans = keys[0]
                    # multiple answers -> sample one answer
                    else:
                        value_sum = sum(values)
                        prob = [value / value_sum for value in values]
                        choice = np.random.multinomial(1, prob).argmax()
                        ans = keys[choice]
                else:
                    ans = -1
            else:
                ans = -1
            out_dict['ans'] = ans

        if self.args.vis_mask_predict:
            if self.args.vis_all_mask:
                if self.args.vis_sampling:
                    grid_size = int(math.sqrt(self.args.n_vis_sampling))
                else:
                    grid_size = self.args.grid_size
                vis_mask = torch.ones(grid_size, grid_size)
                out_dict['vis_mask'] = vis_mask.flatten()

            else:
                if self.args.square_mask:
                    if self.args.vis_sampling:
                        grid_size = int(math.sqrt(self.args.n_vis_sampling))
                    else:
                        grid_size = self.args.grid_size
                    mask_size = random.randint(1, grid_size)
                    vis_mask = torch.zeros(grid_size, grid_size)
                    mask_position_h = random.randint(0, grid_size - mask_size)
                    mask_position_w = random.randint(0, grid_size - mask_size)
                    vis_mask[mask_position_h:mask_position_h + mask_size, mask_position_w:mask_position_w + mask_size] = 1
                    out_dict['vis_mask'] = vis_mask.flatten()

                else:
                    if self.args.vis_sampling:
                        total_idx = list(range(self.args.n_vis_sampling))
                        n_max_mask = self.args.n_vis_sampling
                    else:
                        if self.args.grid_model:
                            total_idx = list(range(self.n_grids))
                            n_max_mask = self.n_grids
                        else:
                            total_idx = list(range(self.args.n_boxes))
                            n_max_mask = self.n_boxes
                    n_masks = random.randint(1, n_max_mask)
                    vis_mask = torch.zeros(n_max_mask)
                    vis_mask_idx = np.random.choice(total_idx, n_masks, replace=False)
                    vis_mask_idx = torch.from_numpy(vis_mask_idx)
                    vis_mask[vis_mask_idx] = 1
                    out_dict['vis_mask'] = vis_mask

                if self.args.VMP_smart:
                    if self.args.square_mask:
                        if self.args.vis_sampling:
                            grid_size = int(math.sqrt(self.args.n_vis_sampling))
                        else:
                            grid_size = self.args.grid_size
                        mask_size = random.randint(1, grid_size)
                        vis_mask = torch.zeros(grid_size, grid_size)
                        mask_position_h = random.randint(0, grid_size - mask_size)
                        mask_position_w = random.randint(0, grid_size - mask_size)
                        vis_mask[mask_position_h:mask_position_h + mask_size, mask_position_w:mask_position_w + mask_size] = 1
                        out_dict['vis_mask_2'] = vis_mask.flatten()

                    else:
                        if self.args.vis_sampling:
                            total_idx = list(range(self.args.n_vis_sampling))
                            n_max_mask = self.args.n_vis_sampling
                        else:
                            if self.args.grid_model:
                                total_idx = list(range(self.n_grids))
                                n_max_mask = self.n_grids
                            else:
                                total_idx = list(range(self.args.n_boxes))
                                n_max_mask = self.n_boxes
                        n_masks = random.randint(1, n_max_mask)
                        vis_mask = torch.zeros(n_max_mask)
                        vis_mask_idx = np.random.choice(total_idx, n_masks, replace=False)
                        vis_mask_idx = torch.from_numpy(vis_mask_idx)
                        vis_mask[vis_mask_idx] = 1
                        out_dict['vis_mask_2'] = vis_mask
        else:
            if self.args.grid_model:
                if self.args.vis_sampling:
                    vis_mask = torch.bernoulli(
                        torch.full((self.args.n_vis_sampling,),  self.args.obj_mask_rate)).bool()
                else:
                    vis_mask = torch.bernoulli(
                        torch.full((self.n_grids,),  self.args.obj_mask_rate)).bool()
                out_dict['vis_mask'] = vis_mask
            else:
                vis_mask = torch.bernoulli(
                    torch.full((self.n_boxes,),  self.args.obj_mask_rate)).bool()
                out_dict['vis_mask'] = vis_mask

        out_dict['args'] = self.args

        if self.args.vis_mask_COCO_only or self.args.vis_mask_COCOVG_only:
            quotient = idx // len(self.COCO_data)
            if len(self.data) - quotient * len(self.COCO_data) < len(self.COCO_data):
                coco_idx = random.randint(0, len(self.COCO_data) - 1)
            else:
                coco_idx = idx % len(self.COCO_data)
            coco_datum = self.COCO_data[coco_idx]

            if self.args.vis_mask_COCO_only:
                assert coco_datum['text_source'] == 'mscoco'
            elif self.args.vis_mask_COCOVG_only:
                assert coco_datum['text_source'] in ['mscoco', 'vg']
            assert 'mscoco' in coco_datum['img_source']

            coco_input_ids, coco_n_tokens = coco_datum['input_ids'], coco_datum['n_tokens']
            coco_input_ids = torch.LongTensor(coco_input_ids)

            out_dict['COCO_input_ids'] = coco_input_ids
            out_dict['COCO_n_tokens'] = coco_n_tokens

            if 'cluster_id' in self.data_out:
                img_id = coco_datum['img_id']
                # cluster_id = datum['cluster_id']
                img_id_to_cluster_id = self.data_source_to_cluster_data[coco_datum['img_source']]
                cluster_id = img_id_to_cluster_id[img_id]
                assert cluster_id is not None, coco_datum

                cluster_id = torch.from_numpy(cluster_id)
                out_dict['COCO_cluster_id'] = cluster_id

        return out_dict


def collate_fn(batch):
    batch_entry = {}

    B = len(batch)
    W_L = max(entry['n_tokens'] for entry in batch)
    V_L = len(batch[0]['boxes'])

    args = batch[0]['args']

    C = 3
    H = W = args.resize_input_size

    word_id = torch.zeros(B, W_L, dtype=torch.long)
    if args.word_mask_predict:
        word_mask = torch.zeros(B, W_L, dtype=torch.bool)

    if args.vis_mask_COCO_only or args.vis_mask_COCOVG_only:
        C_W_L = W_L = max(entry['COCO_n_tokens'] for entry in batch)
        COCO_word_id = torch.zeros(B, C_W_L, dtype=torch.long)

    if args.task_matched:
        O_W_L = max(max(entry['other_n_tokens'] for entry in batch), W_L)
        other_word_id = torch.zeros(B, O_W_L, dtype=torch.long)

    # img = torch.zeros(B, C, H, W, dtype=torch.float)
    box_position = torch.zeros(B, V_L, 4, dtype=torch.float)
    vis_mask = torch.zeros(B, V_L, dtype=torch.bool)
    if args.VMP_smart:
        vis_mask_2 = torch.zeros(B, V_L, dtype=torch.bool)

    matched_label = torch.zeros(B, dtype=torch.long)

    if 'cluster_id' in args.data_out:
        cluster_id = torch.zeros(B, V_L, dtype=torch.long)
        if args.vis_mask_COCO_only or args.vis_mask_COCOVG_only:
            COCO_cluster_id = torch.zeros(B, V_L, dtype=torch.long)
    if 'feat' in args.data_out:
        vis_feats = torch.zeros(B, V_L, args.feat_dim, dtype=torch.float)
    if 'prob' in args.data_out:
        probs = torch.zeros(B, V_L, args.n_class, dtype=torch.float)
    if 'obj_prob' in args.data_out:
        obj_probs = torch.zeros(B, V_L, 1600, dtype=torch.float)
    if 'attr_prob' in args.data_out:
        attr_probs = torch.zeros(B, V_L, 400, dtype=torch.float)
    if 'obj_id' in args.data_out:
        obj_ids = torch.zeros(B, V_L, dtype=torch.long)

    img_path_list = []
    sentences = []
    ans = []
    uids = []

    for i, entry in enumerate(batch):
        word_id[i, :entry['n_tokens']] += entry['input_ids']
        if args.word_mask_predict:
            word_mask[i][entry['word_mask_idx']] = 1

        if args.vis_mask_COCO_only or args.vis_mask_COCOVG_only :
            COCO_word_id[i, :entry['COCO_n_tokens']] += entry['COCO_input_ids']

        if args.task_matched:
            if entry['matched_label'] == 0:
                other_word_id[i, :entry['other_n_tokens']] += entry['other_input_ids']
            elif entry['matched_label'] == 1:
                other_word_id[i, :entry['n_tokens']] += entry['input_ids']

        sentences.append(entry['sent'])

        # img_path_list.append(entry['img_path'])
        # img[i] += entry['img']
        # if args.vis_mask_predict:
        #     vis_mask[i] += entry['vis_mask'].bool()
        #     if args.VMP_smart:
        #         vis_mask_2[i] += entry['vis_mask_2'].bool()
        # if args.vis_mask_predict:
        vis_mask[i] += entry['vis_mask'].bool()
        if args.VMP_smart:
            vis_mask_2[i] += entry['vis_mask_2'].bool()
        box_position[i] += entry['boxes']

        matched_label[i] += entry['matched_label']

        if 'cluster_id' in args.data_out:
            cluster_id[i] += entry['cluster_id']
            if args.vis_mask_COCO_only or args.vis_mask_COCOVG_only:
                COCO_cluster_id[i] += entry['COCO_cluster_id']
        if 'feat' in args.data_out:
            vis_feats[i] += entry['vis_feats']
        if 'prob' in args.data_out:
            probs[i] += entry['probs']

        if 'obj_prob' in args.data_out:
            obj_probs[i] += entry['obj_probs']
        if 'attr_prob' in args.data_out:
            attr_probs[i] += entry['attr_probs']

        if 'obj_id' in args.data_out:
            obj_ids[i] += entry['obj_id']

        if 'ans' in args.data_out:
            ans.append(entry['ans'])

        uids.append(entry['uid'])

    batch_entry['word_id'] = word_id

    if args.word_mask_predict:
        word_labels = word_id.clone().detach()
        word_labels[~word_mask] = -1
        masked_word_id = word_id.clone().detach()
        masked_word_id[word_mask] = tokenizer.vocab["[MASK]"]
    elif args.task_mask_lm:
        masked_word_id, word_labels = random_word_batch(word_id, args.word_mask_rate)

    if args.task_mask_lm or args.word_mask_predict:
        batch_entry['masked_word_id'] = masked_word_id
        batch_entry['word_label'] = word_labels

    if args.task_matched:
        batch_entry['other_word_id'] = other_word_id

    batch_entry['box_position'] = box_position
    batch_entry['vis_mask'] = vis_mask
    if args.VMP_smart:
        batch_entry['vis_mask_2'] = vis_mask_2

    batch_entry['matched_label'] = matched_label

    if 'cluster_id' in args.data_out:
        batch_entry['cluster_id'] = cluster_id
    if 'feat' in args.data_out:
        batch_entry['vis_feats'] = vis_feats

    if 'obj_prob' in args.data_out:
        batch_entry['obj_label'] = obj_probs.argmax(dim=-1)
    if 'attr_prob' in args.data_out:
        batch_entry['attr_label'] = attr_probs.argmax(dim=-1)

    if 'obj_id' in args.data_out:
        batch_entry['obj_ids'] = obj_ids

    if args.vis_mask_COCO_only or args.vis_mask_COCOVG_only:
        batch_entry['COCO_word_id'] = COCO_word_id
        if 'cluster_id' in args.data_out:
            batch_entry['COCO_cluster_id'] = COCO_cluster_id

    if args.task_qa:
        batch_entry['qa_label'] = torch.LongTensor(ans)

    # batch_entry['img'] = img
    # batch_entry['img_path'] = img_path_list
    batch_entry['sent'] = sentences
    batch_entry['uid'] = uids

    return batch_entry


def get_loader(args, split='mscoco_train', mode='train',
               batch_size=32, workers=4, distributed=False, gpu=0,
               collate_fn=collate_fn, transform=None, topk=None, data_out=['img']):

    if transform is None:
        resize_target_size = 160  # 32 * 5
        resize = transforms.Resize((resize_target_size, resize_target_size))
        hflip = transforms.RandomHorizontalFlip()

        transform = transforms.Compose([
            resize,
            hflip
        ])

    if 'mscoco' in split:
        verbose = gpu == 0
        dataset = COCODataset(split, transform=transform,
                              topk=topk, data_out=data_out, verbose=verbose, args=args)
    # elif 'CC' in split:
    #     dataset = CCDataset(split, transform=transform, topk=topk)

    if distributed and mode == 'train':
        # sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank)
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
            collate_fn=collate_fn)

    return loader


def box_position(grid_size=5):
    n_grids = grid_size ** 2
    boxes = np.zeros(shape=(n_grids, 4), dtype=np.float32)
    for i in range(grid_size):
        for j in range(grid_size):
            # pre-normalize (0 ~ 1)
            x0, x1 = j / grid_size, (j + 1) / grid_size
            y0, y1 = i / grid_size, (i + 1) / grid_size
            coordinate = (x0, y0, x1, y1)
            boxes[i * grid_size + j] = coordinate
    return boxes


def random_word_batch(inputs, mlm_probability=0.15):
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    labels = inputs.clone().detach()
    masked_inputs = inputs.clone().detach()
    # We sample a few tokens in each sequence for masked-LM training (with probability mlm_probability defaults to 0.15 in Bert/RoBERTa)

    masked_indices = torch.bernoulli(
        torch.full(labels.shape, mlm_probability)).bool()
    # do not mask special tokens
    masked_indices[:, 0] = False
    masked_indices[:, -1] = False

    labels[~masked_indices] = -1  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(
        labels.shape, 0.8)).bool() & masked_indices
    masked_inputs[indices_replaced] = tokenizer.vocab["[MASK]"]

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(
        labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(
        len(tokenizer.vocab), labels.shape, dtype=torch.long)
    masked_inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return masked_inputs, labels


def random_feat_batch(feats, mask_rate=0.30, clustering=True, n_centroids=30000, vis_mask=None):
    if clustering:
        masked_feat_ids = feats.clone().detach()
        VIS_MASK_ID = -100

        # BERT-style
        B, n_grids = masked_feat_ids.size()
        vis_mask = torch.bernoulli(
            torch.full((B, n_grids),  mask_rate)).bool()

        # 80% => [MASK]
        indices_mask = torch.bernoulli(torch.full(
            (B, n_grids), 0.8)).bool() & vis_mask
        masked_feat_ids[indices_mask] = VIS_MASK_ID

        # 10% => random cluster
        indices_random = torch.bernoulli(torch.full(
            (B, n_grids), 0.5)).bool() & vis_mask & ~indices_mask
        random_ids = torch.randint(
            n_centroids, (B, n_grids), dtype=torch.long)
        masked_feat_ids[indices_random] = random_ids[indices_random]

        # 10% => no change
        return masked_feat_ids, vis_mask

    else:
        masked_feats = feats.clone().detach()

        B, n_grids, z_dim = masked_feats.size()

        vis_mask = torch.bernoulli(
            torch.full((B, n_grids, 1), mask_rate)).bool()

        mask_value = 0
        masked_feats.masked_fill_(vis_mask, mask_value)

        return masked_feats, vis_mask


def make_uid(img_id, dset, sent_idx):
    return "%s_%s_%03d" % (img_id, dset, sent_idx)

class QAEvaluator:
    def __init__(self, data):

        # Create QA Eval Data
        self.data = []
        for datum in data:
            sentf = datum['sentf']
            for sents_cat, sents in sentf.items():
                if sents_cat in datum['labelf']:    # A labeled dataset
                    labels = datum['labelf'][sents_cat]
                    for sent_idx, sent in enumerate(sents):
                        new_datum = {
                            'uid': make_uid(datum['img_id'], sents_cat, sent_idx),
                            'img_id': datum['img_id'],
                            'sent': sent,
                            'dset': sents_cat,
                            'label': labels[sent_idx]
                        }
                        self.data.append(new_datum)

        # uid2datum
        self.uid2datum = {}
        for datum in self.data:
            self.uid2datum[datum['uid']] = datum

    def evaluate(self, uid2ans: dict):
        score = 0.
        cnt = 0
        dset2score = defaultdict(lambda: 0.)
        dset2cnt = defaultdict(lambda: 0)
        for uid, ans in uid2ans.items():
            if uid not in self.uid2datum:   # Not a labeled data
                continue
            datum = self.uid2datum[uid]
            label = datum['label']
            dset = datum['dset']
            if ans in label:
                score += label[ans]
                dset2score[dset] += label[ans]
            cnt += 1
            dset2cnt[dset] += 1
        return dset2score, dset2cnt, score, cnt

    def _evaluate(self, uid2ans: dict, pprint=False):
        score = 0.
        cnt = 0
        dset2score = defaultdict(lambda: 0.)
        dset2cnt = defaultdict(lambda: 0)
        for uid, ans in uid2ans.items():
            if uid not in self.uid2datum:   # Not a labeled data
                continue
            datum = self.uid2datum[uid]
            label = datum['label']
            dset = datum['dset']
            if ans in label:
                score += label[ans]
                dset2score[dset] += label[ans]
            cnt += 1
            dset2cnt[dset] += 1
        accu = score / cnt
        dset2accu = {}
        for dset in dset2cnt:
            dset2accu[dset] = dset2score[dset] / dset2cnt[dset]

        if pprint:
            accu_str = "Overall Accu %0.4f, " % (accu)
            sorted_keys = sorted(dset2accu.keys())
            for key in sorted_keys:
                accu_str += "%s Accu %0.4f, " % (key, dset2accu[key])
            print(accu_str)

        return accu, dset2accu

    def dump_result(self, uid2ans: dict, path):
        raise NotImplemented
