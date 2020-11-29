# coding=utf-8
# Copyleft 2019 project LXRT.

from pathlib import Path
# from collections import defaultdict
import json
import random
from multiprocessing import Pool
import h5py
import pickle

from tqdm import tqdm
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from PIL import Image
from torchvision.datasets.folder import default_loader

from lxrt.tokenization import BertTokenizer
from utils import box_position


# from param import args
# feat_dim = args.feat_dim
# n_class = args.n_class
# n_centroids = args.n_centroids
# n_iter = args.n_iter
# cluster_src = args.cluster_src
# from resnet_feat_config import feat_dim, n_class, n_centroids, n_iter

tokenizer = BertTokenizer.from_pretrained(
    "bert-base-uncased",
    do_lower_case=True
)

# assert tokenizer.convert_tokens_to_ids(['[PAD]'])[0] == 0

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
vg_img_dir = Path('/home/jaeminc/workspace/datasets/VG/').resolve()
# sent_sources = ['mscoco', 'vg']
sent_sources = ['mscoco']



def get_datum(info_dict):
    _data = []
    _sents = []
    for dataset, sents in info_dict['sentf'].items():
        if dataset not in sent_sources:
            continue

        img_id = info_dict['img_id']

        # if info_dict['clustering']:
        #     cluster_id = None
        #     if info_dict['source'] == 'mscoco_train':
        #         # global mscoco_train_img_id_to_cluster_id
        #         cluster_id = mscoco_train_img_id_to_cluster_id[img_id]
        #         # img_id_to_cluster_id = mscoco_train_img_id_to_cluster_id
        #         # assert img_id_to_cluster_id is not None
        #     elif info_dict['source'] in ['mscoco_minival', 'mscoco_nominival']:
        #         # global mscoco_valid_img_id_to_cluster_id
        #         cluster_id = mscoco_valid_img_id_to_cluster_id[img_id]
        #         # img_id_to_cluster_id = mscoco_valid_img_id_to_cluster_id
        #         # assert img_id_to_cluster_id is not None
        #     elif info_dict['source'] == 'vgnococo':
        #         # global vg_img_id_to_cluster_id
        #         cluster_id = vg_img_id_to_cluster_id[img_id]
        #         # img_id_to_cluster_id = vg_img_id_to_cluster_id
        #         # assert img_id_to_cluster_id is not None
        #
        #     # try:
        #     #     cluster_id = img_id_to_cluster_id[img_id]
        #     # except TypeError:
        #     #     print(info_dict)
        #     #     exit()
        #     assert cluster_id is not None, info_dict

        img_path = None

        if 'COCO_val' in img_id:
            img_dir = coco_img_dir.joinpath('val2014')
            if img_dir.joinpath(img_id + '.jpg').is_file():
                img_path = img_dir.joinpath(img_id + '.jpg')
            elif img_dir.joinpath(img_id + '.png').is_file():
                img_path = img_dir.joinpath(img_id + '.png')
        elif 'COCO_train' in img_id:
            img_dir = coco_img_dir.joinpath('train2014')
            if img_dir.joinpath(img_id + '.jpg').is_file():
                img_path = img_dir.joinpath(img_id + '.jpg')
            elif img_dir.joinpath(img_id + '.png').is_file():
                img_path = img_dir.joinpath(img_id + '.png')
        else:
            img_dir = vg_img_dir
            if img_dir.joinpath('VG_100K', img_id + '.jpg').is_file():
                img_path = img_dir.joinpath('VG_100K', img_id + '.jpg')
            elif img_dir.joinpath('VG_100K_2', img_id + '.jpg').is_file():
                img_path = img_dir.joinpath('VG_100K_2', img_id + '.jpg')

            try:
                with Image.open(img_path) as im:
                    pass
            except IOError:
                continue

        assert img_path is not None, img_id

        for sent_id, sent in enumerate(sents):
            # remove duplicate sentence
            if sent not in _sents:

                sent = sent.strip()

                input_ids, n_tokens = text_process(sent, max_text_length=info_dict['max_text_length'])

                # if n_tokens <= 2:
                #     continue

                datum = {
                    'uid': f'{img_id}_{dataset}_{sent_id}',
                    'source': info_dict['source'],
                    'img_id': img_id,
                    'img_path': img_path,
                    # 'img': img,
                    'sent': sent,
                    'input_ids': input_ids,
                    'n_tokens': n_tokens,
                    # 'cluster_id': cluster_id
                }
                _data.append(datum)

            break


            _sents.append(sent)
    return _data


class COCODataset(Dataset):
    def __init__(self, split='mscoco_mininval', coco_cap_only=True, image_only=True,
                 loader=default_loader, transform=None, topk=-1,
                 data_out=['img'], verbose=True, args=None):

        self.loader = loader
        self.transform = transform
        self.data_out = data_out
        self.topk = topk
        self.verbose = verbose
        self.args = args

        # Loading datasets to data
        self.sources = split.split(',')
        if self.verbose:
            print('Data sources: ', self.sources)
        data_info_dicts = []

        self.img_ids_to_source = {}

        for source in self.sources:
            data_info_path = project_dir.joinpath(f'data/lxmert/{source}.json')
            with open(data_info_path) as f:
                _data_info_dicts = json.load(f)
                # source_img_ids.append([d['img_id'] for d in _data_info_dicts])
                for _d in _data_info_dicts:
                    self.img_ids_to_source[_d['img_id']] = source
                    _d['source'] = source
                    _d['clustering'] = args.clustering
                    _d['max_text_length'] = args.max_text_length
                data_info_dicts.extend(_data_info_dicts)
            if self.verbose:
                print(f"Loaded {len(_data_info_dicts)} data from", source)

        if self.topk > 0:
            data_info_dicts = data_info_dicts[:self.topk]
            if self.verbose:
                print(f"Use only {self.topk} data")

        if args.clustering:
            centroids_dir = Path(
                '/home/jaeminc/Dropbox/Projects/AI2/clustering/').resolve()
            if args.v4:
                centroids_dir = centroids_dir.joinpath('v4')
            imsize = args.resize_input_size

            if args.im_ratio == 'original':
                if 'mscoco_train' in split:
                    with open(centroids_dir.joinpath(f'{args.encoder}_{args.cluster_src}_mscoco_train_img_id_to_cluster_id_{args.n_centroids}_iter{args.n_iter}_d{args.feat_dim}_grid{args.grid_size}.pkl'), 'rb') as f:
                        mscoco_train_img_id_to_cluster_id = pickle.load(f)
                    # self.img_id_to_cluster_id = mscoco_train_img_id_to_cluster_id
                if 'mscoco_minival' in split or 'mscoco_nominival' in split:
                    with open(centroids_dir.joinpath(f'{args.encoder}_{args.cluster_src}_mscoco_valid_img_id_to_cluster_id_{args.n_centroids}_iter{args.n_iter}_d{args.feat_dim}_grid{args.grid_size}.pkl'), 'rb') as f:
                        mscoco_valid_img_id_to_cluster_id = pickle.load(f)
                    # self.img_id_to_cluster_id = mscoco_valid_img_id_to_cluster_id
            else:
                if 'mscoco_train' in split:
                    with open(centroids_dir.joinpath(f'{args.encoder}_{args.cluster_src}_mscoco_train_img_id_to_cluster_id_{args.n_centroids}_iter{args.n_iter}_d{args.feat_dim}_grid{args.grid_size}_imsize{imsize}.pkl'), 'rb') as f:
                        mscoco_train_img_id_to_cluster_id = pickle.load(f)
                    # self.img_id_to_cluster_id = mscoco_train_img_id_to_cluster_id
                if 'mscoco_minival' in split or 'mscoco_nominival' in split:
                    with open(centroids_dir.joinpath(f'{args.encoder}_{args.cluster_src}_mscoco_valid_img_id_to_cluster_id_{args.n_centroids}_iter{args.n_iter}_d{args.feat_dim}_grid{args.grid_size}_imsize{imsize}.pkl'), 'rb') as f:
                        mscoco_valid_img_id_to_cluster_id = pickle.load(f)
                    # self.img_id_to_cluster_id = mscoco_valid_img_id_to_cluster_id

            self.data_source_to_cluster_data = {}
            if 'mscoco_train' in split:
                self.data_source_to_cluster_data['mscoco_train'] = mscoco_train_img_id_to_cluster_id
            if 'mscoco_minival' in split:
                self.data_source_to_cluster_data['mscoco_minival'] = mscoco_valid_img_id_to_cluster_id
            if 'mscoco_nominival' in split:
                self.data_source_to_cluster_data['mscoco_nominival'] = mscoco_valid_img_id_to_cluster_id

            # global mscoco_train_img_id_to_cluster_id
            # with open(centroids_dir.joinpath(f'{args.encoder}_{args.cluster_src}_mscoco_train_img_id_to_cluster_id_{args.n_centroids}_iter{args.n_iter}_d{args.feat_dim}_grid{args.grid_size}_imsize{imsize}.pkl'), 'rb') as f:
            #     mscoco_train_img_id_to_cluster_id = pickle.load(f)
            # # global mscoco_valid_img_id_to_cluster_id
            # with open(centroids_dir.joinpath(f'{args.encoder}_{args.cluster_src}_mscoco_valid_img_id_to_cluster_id_{args.n_centroids}_iter{args.n_iter}_d{args.feat_dim}_grid{args.grid_size}_imsize{imsize}.pkl'), 'rb') as f:
            #     mscoco_valid_img_id_to_cluster_id = pickle.load(f)
            # # global vg_img_id_to_cluster_id
            # with open(centroids_dir.joinpath(f'{args.encoder}_{args.cluster_src}_vg_img_id_to_cluster_id_{args.n_centroids}_iter{args.n_iter}_d{args.feat_dim}_grid{args.grid_size}_imsize{imsize}.pkl'), 'rb') as f:
            #     vg_img_id_to_cluster_id = pickle.load(f)

            # self.data_source_to_cluster_data = {
            #     'mscoco_train': mscoco_train_img_id_to_cluster_id,
            #     'mscoco_minival': mscoco_valid_img_id_to_cluster_id,
            #     'mscoco_nominival': mscoco_valid_img_id_to_cluster_id,
            #     'vg': vg_img_id_to_cluster_id
            # }
        #
        #     assert mscoco_train_img_id_to_cluster_id is not None
        #     assert mscoco_valid_img_id_to_cluster_id is not None
        #     assert vg_img_id_to_cluster_id is not None
        #
        #     if self.verbose:
        #         print('centroids loaded')

        data = []
        with Pool(8) as pool:
            if self.verbose:
                for _data in tqdm(pool.imap(get_datum, data_info_dicts),
                                  total=len(data_info_dicts), ncols=150):
                    data.extend(_data)
            else:
                for _data in pool.imap(get_datum, data_info_dicts):
                    data.extend(_data)

        if 'feat' in self.data_out:
            # dataset_dir = Path('/home/jaeminc/workspace/datasets').resolve()
            dataset_dir = Path('/net/nfs2.corp/prior/jaeminc/datasets').resolve()
            if args.grid_model:
                if args.v4:
                    if args.im_ratio == 'original':
                        self.data_source_to_h5_path = {
                            'mscoco_train': dataset_dir.joinpath(f'COCO/features/{args.encoder}_train_v4_grid{args.grid_size}.h5'),
                            'mscoco_minival': dataset_dir.joinpath(f'COCO/features/{args.encoder}_valid_v4_grid{args.grid_size}.h5'),
                            'mscoco_nominival': dataset_dir.joinpath(f'COCO/features/{args.encoder}_valid_v4_grid{args.grid_size}.h5'),
                            # 'vgnococo': dataset_dir.joinpath(f'VG/features/{args.encoder}_v4_grid{args.grid_size}.h5'),
                        }
                    else:
                        self.data_source_to_h5_path = {
                            'msoco_train': dataset_dir.joinpath(f'COCO/features/{args.encoder}_train_v4_grid{args.grid_size}_imsize{args.imsize}.h5'),
                            'msoco_minival': dataset_dir.joinpath(f'COCO/features/{args.encoder}_valid_v4_grid{args.grid_size}_imsize{args.imsize}.h5'),
                            'msoco_nominival': dataset_dir.joinpath(f'COCO/features/{args.encoder}_valid_v4_grid{args.grid_size}_imsize{args.imsize}.h5'),
                            # 'vgnococo': dataset_dir.joinpath(f'VG/features/{args.encoder}_v4_grid{args.grid_size}_imsize{args.imsize}.h5'),
                        }

            else:
                self.data_source_to_h5_path = {
                    # 'mscoco_train': dataset_dir.joinpath(f'COCO/features/maskrcnn_train_boxes36.h5'),
                    'mscoco_minival': dataset_dir.joinpath(f'COCO/features/maskrcnn_valid_boxes36.h5'),
                    'mscoco_nominival': dataset_dir.joinpath(f'COCO/features/maskrcnn_valid_boxes36.h5'),
                    # 'vgnococo': dataset_dir.joinpath(f'VG/features/maskrcnn_boxes36.h5'),
                }

            for source, path in self.data_source_to_h5_path.items():
                assert path.is_file(), (source, path)

            self.source_to_h5 = None

        self.data = data
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
        self.tasks = tasks

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        out_dict = {}

        datum = self.data[idx]

        ###### Image ######
        img_id = datum['img_id']
        out_dict['img_id'] = img_id
        out_dict['uid'] = datum['uid']

        if 'feat' in self.data_out:
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
            np.testing.assert_array_less(-boxes, 0+1e-5)
            boxes = torch.from_numpy(boxes)

            boxes.clamp_(min=0.0, max=1.0)

        out_dict['boxes'] = boxes

        ###### Text #####
        sent = datum['sent']
        # input_ids, n_tokens = text_process(sent)
        input_ids, n_tokens = datum['input_ids'], datum['n_tokens']
        input_ids = torch.LongTensor(input_ids)

        out_dict['sent'] = sent
        out_dict['input_ids'] = input_ids
        out_dict['n_tokens'] = n_tokens

        # Flip -> Img-Text not matched
        if self.args.task_matched and random.random() < 0.5:
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

        if 'cluster_id' in self.data_out:

            img_id_to_cluster_id = self.data_source_to_cluster_data[datum['source']]
            cluster_id = img_id_to_cluster_id[img_id]
            assert cluster_id is not None, datum

            # cluster_id = datum['cluster_id']
            cluster_id = torch.from_numpy(cluster_id)
            out_dict['cluster_id'] = cluster_id
        else:
            cluster_id = None

        if 'img' in self.data_out:
            # img_id = datum['img_id']
            img_path = datum['img_path']
            img = self.loader(img_path)
            img = self.transform(img)

            out_dict['img'] = img
            out_dict['img_path'] = img_path

        if self.args.word_mask_predict:
            total_idx = list(range(1, 1 + n_tokens))  # Don't mask CLS / SEP
            n_masks = random.randint(1, n_tokens)
            word_mask_idx = np.random.choice(total_idx, n_masks, replace=False)
            word_mask_idx = torch.from_numpy(word_mask_idx)

            out_dict['word_mask_idx'] = word_mask_idx

        if self.args.vis_mask_predict:
            if self.args.vis_all_mask:
                grid_size = self.args.grid_size
                vis_mask = torch.ones(grid_size, grid_size)
                out_dict['vis_mask'] = vis_mask.flatten()

            else:
                if self.args.square_mask:
                    grid_size = self.args.grid_size
                    mask_size = random.randint(1, grid_size)
                    vis_mask = torch.zeros(grid_size, grid_size)
                    mask_position_h = random.randint(0, grid_size - mask_size)
                    mask_position_w = random.randint(0, grid_size - mask_size)
                    vis_mask[mask_position_h:mask_position_h + mask_size, mask_position_w:mask_position_w + mask_size] = 1
                    out_dict['vis_mask'] = vis_mask.flatten()

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
                        grid_size = self.args.grid_size
                        mask_size = random.randint(1, grid_size)
                        vis_mask = torch.zeros(grid_size, grid_size)
                        mask_position_h = random.randint(0, grid_size - mask_size)
                        mask_position_w = random.randint(0, grid_size - mask_size)
                        vis_mask[mask_position_h:mask_position_h + mask_size, mask_position_w:mask_position_w + mask_size] = 1
                        out_dict['vis_mask_2'] = vis_mask.flatten()

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
                vis_mask = torch.bernoulli(
                    torch.full((self.n_grids,),  self.args.obj_mask_rate)).bool()
                out_dict['vis_mask'] = vis_mask

        out_dict['args'] = self.args

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

    if args.task_matched:
        O_W_L = max(max(entry['other_n_tokens'] for entry in batch), W_L)
        other_word_id = torch.zeros(B, O_W_L, dtype=torch.long)

    img = torch.zeros(B, C, H, W, dtype=torch.float)
    boxes = torch.zeros(B, V_L, 4, dtype=torch.float)
    vis_mask = torch.zeros(B, V_L, dtype=torch.bool)
    if args.VMP_smart:
        vis_mask_2 = torch.zeros(B, V_L, dtype=torch.bool)

    matched_label = torch.zeros(B, dtype=torch.long)

    if args.clustering:
        cluster_id = torch.zeros(B, V_L, dtype=torch.long)

    if 'feat' in args.data_out:
        vis_feats = torch.zeros(B, V_L, args.feat_dim, dtype=torch.float)

    img_path_list = []
    sentences = []

    img_ids = []

    uid_list = []

    for i, entry in enumerate(batch):

        img_ids.append(entry['img_id'])
        uid_list.append(entry['uid'])

        word_id[i, :entry['n_tokens']] += entry['input_ids']
        if args.word_mask_predict:
            word_mask[i][entry['word_mask_idx']] = 1

        if args.task_matched:
            if entry['matched_label'] == 0:
                other_word_id[i, :entry['other_n_tokens']] += entry['other_input_ids']
            elif entry['matched_label'] == 1:
                other_word_id[i, :entry['n_tokens']] += entry['input_ids']

        sentences.append(entry['sent'])

        img_path_list.append(entry['img_path'])
        img[i] += entry['img']
        # if args.vis_mask_predict:
        #     vis_mask[i] += entry['vis_mask'].bool()
        #     if args.VMP_smart:
        #         vis_mask_2[i] += entry['vis_mask_2'].bool()
        vis_mask[i] += entry['vis_mask'].bool()
        if args.VMP_smart:
            vis_mask_2[i] += entry['vis_mask_2'].bool()
        boxes[i] += entry['boxes']

        if 'feat' in args.data_out:
            vis_feats[i] += entry['vis_feats']

        matched_label[i] += entry['matched_label']

        if args.clustering:
            cluster_id[i] += entry['cluster_id']

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
        batch_entry['word_labels'] = word_labels

    if args.task_matched:
        batch_entry['other_word_id'] = other_word_id

    batch_entry['img'] = img
    batch_entry['img_id'] = img_ids
    batch_entry['boxes'] = boxes
    batch_entry['vis_mask'] = vis_mask
    if args.VMP_smart:
        batch_entry['vis_mask_2'] = vis_mask_2

    batch_entry['matched_label'] = matched_label

    if args.clustering:
        batch_entry['cluster_id'] = cluster_id

    if 'feat' in args.data_out:
        batch_entry['vis_feats'] = vis_feats

    batch_entry['img_path'] = img_path_list
    batch_entry['sent'] = sentences
    batch_entry['uid'] = uid_list

    return batch_entry


def get_loader(args, split='mscoco_train', mode='train',
               batch_size=32, workers=4, distributed=False, gpu=0,
               collate_fn=collate_fn, transform=None, topk=-1,
               data_out=['img']):

    if 'mscoco' in split:
        verbose = gpu == 0
        dataset = COCODataset(split, transform=transform,
                              topk=topk, data_out=data_out, verbose=verbose, args=args)
    # elif 'CC' in split:
    #     dataset = CCDataset(split, transform=transform, topk=topk)

    if distributed:
        world_size = args.world_size
        local_rank = gpu
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank)
    else:
        sampler = None

    if mode == 'train':
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=(sampler is None),
            num_workers=workers, pin_memory=True, sampler=sampler,
            collate_fn=collate_fn)
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=True,
            sampler=sampler,
            collate_fn=collate_fn)

    return loader


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


# def random_feat_batch(cluster_ids, mask_rate=0.15, clustering=True, n_centroids=10000, vis_mask=None):
#
#     masked_cluster_ids = cluster_ids.clone().detach()
#     VIS_MASK_ID = -100
#
#     # BERT-style
#     B, n_grids = masked_cluster_ids.size()
#     vis_mask = torch.bernoulli(
#         torch.full((B, n_grids),  mask_rate)).bool()
#
#     # 80% => [MASK]
#     indices_mask = torch.bernoulli(torch.full(
#         (B, n_grids), 0.8)).bool() & vis_mask
#     masked_cluster_ids[indices_mask] = VIS_MASK_ID
#
#     # 10% => random cluster
#     indices_random = torch.bernoulli(torch.full(
#         (B, n_grids), 0.5)).bool() & vis_mask & ~indices_mask
#     random_ids = torch.randint(
#         n_centroids, (B, n_grids), dtype=torch.long)
#     masked_cluster_ids[indices_random] = random_ids[indices_random]
#
#     # 10% => no change
#     return masked_cluster_ids, vis_mask
