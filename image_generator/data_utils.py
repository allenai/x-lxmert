
import pickle
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path
import jsonlines
import h5py
from tqdm import tqdm
import json
from torchvision import transforms
from torchvision.datasets.folder import default_loader

from multiprocessing import Pool

project_dir = Path(__file__).resolve().parent


def grid_view(tensor, n_grid=5):
    """[B, C, H, W]
    => [B * n_grid * n_grid, C, patch_size, patch_size]"""
    B, C, H, W = tensor.size()
    assert H == W
    patch_size = H // n_grid
    assert H % n_grid == 0

    tensor = tensor.view(B, C, n_grid, patch_size, n_grid, patch_size)
    # [B, n_grid, n_grid, C, patch_size, patch_size]
    tensor = tensor.permute(0, 2, 4, 1, 3, 5)
    tensor = tensor.contiguous()
    tensor = tensor.view(B * n_grid * n_grid, C, patch_size, patch_size)
    return tensor


def original_view(tensor, n_grid=5):
    """[B * n_grid * n_grid, C, patch_size, patch_size]
    => [B, C, H, W]"""
    B, C, patch_size, _ = tensor.size()
    assert patch_size == _
    H = W = patch_size * n_grid
    B = B // (n_grid ** 2)
    tensor = tensor.view(B, n_grid, n_grid, C, patch_size, patch_size)
    tensor = tensor.permute(0, 3, 1, 4, 2, 5)
    tensor = tensor.contiguous()
    tensor = tensor.view(B, C, H, W)
    return tensor


def box_position(n_grid=5):
    boxes = np.zeros(shape=(n_grid, n_grid, 4), dtype=np.float32)
    for i in range(n_grid):
        for j in range(n_grid):
            x0, x1 = j, j + 1
            y0, y1 = i, i + 1
            coordinate = (x0, y0, x1, y1)
            boxes[i, j] = coordinate
    # (0 ~ n_grid) => (0 ~ 1)
    boxes = boxes / n_grid
    return boxes


# transform = transforms.Compose([
#     # transforms.Resize((112, 112)),
#     transforms.Resize((224, 224)),
#     # transforms.RandomHorizontalFlip()
# ])

coco_dir = Path('../datasets/COCO').resolve()
coco_img_dir = coco_dir.joinpath('images/').resolve()


def get_datum(info_dict):
    # resize_target_size = info_dict['target_size']
    img_id = info_dict['img_id']

    if 'val' in img_id:
        img_dir = coco_img_dir.joinpath('val2014').resolve()
    else:
        img_dir = coco_img_dir.joinpath('train2014').resolve()

    img_path = None
    if img_dir.joinpath(img_id + '.jpg').is_file():
        img_path = img_dir.joinpath(img_id + '.jpg')
    elif coco_img_dir.joinpath(img_id + '.png').is_file():
        img_path = img_dir.joinpath(img_id + '.png')
    assert img_path is not None

    # img = default_loader(img_path)
    # img = transform(img)

    # img = img.resize((resize_target_size, resize_target_size), Image.LANCZOS)


    datum = {
        'img_id': img_id,
        'img_path': img_path,
        # 'img': img,
    }

    return datum


class COCODataset(Dataset):
    def __init__(self, config, split='mscoco_mininval', coco_cap_only=True, image_only=True,
                 loader=default_loader, transform=None, topk=-1, data_out=['img'], verbose=True):

        self.config = config
        self.loader = loader
        self.transform = transform
        self.data_out = data_out
        self.topk = topk
        self.verbose = verbose

        if split == 'mscoco_train':
            img_dir = coco_img_dir.joinpath('train2014')
        elif split == 'mscoco_minival':
            img_dir = coco_img_dir.joinpath('val2014')
        elif split == 'mscoco_nominival':
            img_dir = coco_img_dir.joinpath('val2014')
        if self.verbose:
            print('# images:', len(list(coco_img_dir.iterdir())))

        data_info_path = Path(f'../x_lxmert/data/mscoco_data/{split}.json')
        with open(data_info_path) as f:
            data_info_dicts = json.load(f)
        if self.verbose:
            print(f"Loaded {len(data_info_dicts)} data from", data_info_path)
        if self.topk > 0:
            data_info_dicts = data_info_dicts[:self.topk]
            if self.verbose:
                print(f"Use only {self.topk} data")

        if 'cluster_id' in data_out:
            centroids_dir = Path('../datasets/cluster_centroids').resolve()
            if config.im_ratio == 'original':
                if split == 'mscoco_train':
                    with open(centroids_dir.joinpath(f'{config.encoder}_{config.cluster_src}_mscoco_train_img_id_to_cluster_id_{config.n_centroids}_iter{config.n_iter}_d{config.emb_dim}_grid{config.n_grid}.pkl'), 'rb') as f:
                        mscoco_train_img_id_to_cluster_id = pickle.load(f)
                    self.img_id_to_cluster_id = mscoco_train_img_id_to_cluster_id
                else:
                    with open(centroids_dir.joinpath(f'{config.encoder}_{config.cluster_src}_mscoco_valid_img_id_to_cluster_id_{config.n_centroids}_iter{config.n_iter}_d{config.emb_dim}_grid{config.n_grid}.pkl'), 'rb') as f:
                        mscoco_valid_img_id_to_cluster_id = pickle.load(f)
                    self.img_id_to_cluster_id = mscoco_valid_img_id_to_cluster_id
            else:
                imsize = config.resize_input_size
                if split == 'mscoco_train':
                    with open(centroids_dir.joinpath(f'{config.encoder}_{config.cluster_src}_mscoco_train_img_id_to_cluster_id_{config.n_centroids}_iter{config.n_iter}_d{config.emb_dim}_grid{config.n_grid}_imsize{imsize}.pkl'), 'rb') as f:
                        mscoco_train_img_id_to_cluster_id = pickle.load(f)
                    self.img_id_to_cluster_id = mscoco_train_img_id_to_cluster_id
                else:
                    with open(centroids_dir.joinpath(f'{config.encoder}_{config.cluster_src}_mscoco_valid_img_id_to_cluster_id_{config.n_centroids}_iter{config.n_iter}_d{config.emb_dim}_grid{config.n_grid}_imsize{imsize}.pkl'), 'rb') as f:
                        mscoco_valid_img_id_to_cluster_id = pickle.load(f)
                    self.img_id_to_cluster_id = mscoco_valid_img_id_to_cluster_id

        if 'feat' in data_out:
            if config.im_ratio == 'original':
                if split == 'mscoco_train':
                    self.h5_path = coco_dir.joinpath(f'features/{config.encoder}_train_grid{config.n_grid}.h5')
                else:
                    self.h5_path = coco_dir.joinpath(f'features/{config.encoder}_valid_grid{config.n_grid}.h5')
            self.f = None

        for info_dict in data_info_dicts:
            info_dict['target_size'] = self.config.resize_target_size

        with Pool() as pool:
            if self.verbose:
                data = list(
                    tqdm(pool.imap(get_datum, data_info_dicts),
                         total=len(data_info_dicts)))
            else:
                data = list(pool.imap(get_datum, data_info_dicts))

        self.data = data
        if self.verbose:
            if 'sent' not in self.data_out:
                print("# all images:", len(self.data))
            else:
                print("# all sentences:", len(self.data))

        self.grid_size = self.config.n_grid

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        datum = self.data[idx]

        img_id = datum['img_id']
        img_path = datum['img_path']
        img = default_loader(img_path)
        # img = self.transform(img)
        # img = datum['img']
        img = self.transform(img)

        nump_array = np.asarray(img, dtype=np.uint8)
        if nump_array.ndim < 3:
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.moveaxis(nump_array, -1, 0)  # [H, W, C] => [C, H, W]
        img = torch.from_numpy(nump_array)

        out_dict = {}
        out_dict['config'] = self.config
        out_dict['img'] = img

        if 'cluster_id' in self.data_out:
            cluster_id = self.img_id_to_cluster_id[img_id].flatten()
            cluster_id = torch.from_numpy(cluster_id)
            out_dict['cluster_id'] = cluster_id
        else:
            out_dict['cluster_id'] = None

        if 'feat' in self.data_out:
            if self.f is None:
                self.f = h5py.File(self.h5_path, 'r')

            feats = np.zeros(
                shape=(self.config.n_grid, self.config.n_grid, self.config.emb_dim), dtype=np.float32)
            self.f[f'{img_id}/features'].read_direct(feats)
            feats = np.reshape(feats, (self.config.n_grid**2, self.config.emb_dim))
            feats = torch.from_numpy(feats)
            out_dict['feat'] = feats
            # else:
            #     feats = np.zeros(
            #         shape=(self.n_boxes, self.config.feat_dim), dtype=np.float32)
            #     f[f'{img_id}/features'].read_direct(feats)
            #     feats = torch.from_numpy(feats)
            # out_dict['featfeats'] = feats
        else:
            out_dict['feat'] = None

        return out_dict


def collate_fn(batch):

    config = batch[0]['config']
    load_cluster_id = batch[0]['cluster_id'] is not None
    load_feat = batch[0]['feat'] is not None
    w = h = config.resize_input_size

    B = len(batch)
    img_batch = torch.zeros((B, 3, h, w), dtype=torch.uint8)
    if load_cluster_id:
        cluster_id_batch = torch.zeros((B, config.n_grid**2), dtype=torch.long)
    if load_feat:
        feat_batch = torch.zeros((B, config.n_grid**2, config.emb_dim), dtype=torch.float)

    for i, datum in enumerate(batch):
        img_batch[i] += datum['img']
        if load_cluster_id:
            cluster_id_batch[i] += datum['cluster_id']
        if load_feat:
            feat_batch[i] += datum['feat']

    batch_out = {}
    batch_out['img'] = img_batch
    if load_cluster_id:
        batch_out['cluster_id'] = cluster_id_batch
    if load_feat:
        batch_out['feat'] = feat_batch

    return batch_out


def get_loader(config, split='mscoco_train', mode='train',
               batch_size=32, workers=0, distributed=False, gpu=0,
               transform=None, topk=None, data_out=['img']):

    if transform is None:
        resize_target_size = config.resize_target_size
        resize = transforms.Resize((resize_target_size, resize_target_size))
        # hflip = transforms.RandomHorizontalFlip()

        transform = transforms.Compose([
            resize,
            # hflip
        ])

    if 'mscoco' in split:
        verbose = gpu == 0
        dataset = COCODataset(config, split, transform=transform,
                              topk=topk, data_out=data_out, verbose=verbose)

    if distributed:
        sampler = DistributedSampler(dataset)
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
