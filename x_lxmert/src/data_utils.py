
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path
import jsonlines
# from PIL import Image
import json
from torchvision import transforms
from torchvision.datasets.folder import default_loader

from lxrt.tokenization import BertTokenizer

project_dir = Path(__file__).resolve().parent.parent.parent

# resize_target_size = 225
resize_target_size = 160  # 32 * 5
resize = transforms.Resize((resize_target_size, resize_target_size))
hflip = transforms.RandomHorizontalFlip()

img_transform = transforms.Compose([
    resize,
    hflip
])

tokenizer = BertTokenizer.from_pretrained(
    "bert-base-uncased",
    do_lower_case=True
)


def denorm(x):
    """(-1, 1) => (0, 1)"""
    out = (x + 1) / 2
    return out.clamp(0, 1)


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids


def convert_sents_to_features(sents, max_text_length=20, tokenizer=tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (i, sent) in enumerate(sents):
        tokens_a = tokenizer.tokenize(sent.strip())

        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_text_length - 2:
            tokens_a = tokens_a[:(max_text_length - 2)]

        # Keep segment id which allows loading BERT-weights.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_text_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_text_length
        assert len(input_mask) == max_text_length
        assert len(segment_ids) == max_text_length

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids))
    return features


def sent_transform(sent, max_text_length=20):
    tokens = tokenizer.tokenize(sent.strip())
    if len(tokens) > max_text_length - 2:
        tokens = tokens[:(max_text_length - 2)]
    tokens = ['[CLS]'] + tokens + ['[SEP]']
    segment_ids = [0] * len(tokens)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    padding = [0] * (max_text_length - len(input_ids))
    input_ids += padding
    input_mask += padding
    segment_ids += padding
    return input_ids, input_mask, segment_ids


# def img_to_nparray(img):
#     array = np.asarray(img, dtype=np.uint8)
#     array = np.moveaxis(array, -1, 0)  # [H, W, C] => [C, H, W]
#     return array


# def crop(img, grid_size=5):
#
#     img_width, img_height = img.size
#     width_delta = img_width // grid_size
#     height_delta = img_height // grid_size
#
#     array = img_to_nparray(img)
#
#     out_list = []
#     for i in range(grid_size):
#         for j in range(grid_size):
#             cropped = array[:,
#                             height_delta * i: height_delta * (i + 1),
#                             width_delta * j: width_delta * (j + 1)]
#
#             out_list.append(cropped)
#     return out_list


def grid_view(tensor, grid_size=5):
    """[B, C, H, W]
    => [B * grid_size * grid_size, C, patch_size, patch_size]"""
    B, C, H, W = tensor.size()
    assert H == W
    patch_size = H // grid_size
    assert H % grid_size == 0

    tensor = tensor.view(B, C, grid_size, patch_size, grid_size, patch_size)
    # [B, grid_size, grid_size, C, patch_size, patch_size]
    tensor = tensor.permute(0, 2, 4, 1, 3, 5)
    tensor = tensor.contiguous()
    tensor = tensor.view(B * grid_size * grid_size, C, patch_size, patch_size)
    return tensor


def original_view(tensor, grid_size=5):
    """[B * grid_size * grid_size, C, patch_size, patch_size]
    => [B, C, H, W]"""
    B, C, patch_size, _ = tensor.size()
    assert patch_size == _
    H = W = patch_size * grid_size
    B = B // (grid_size ** 2)
    tensor = tensor.view(B, grid_size, grid_size, C, patch_size, patch_size)
    tensor = tensor.permute(0, 3, 1, 4, 2, 5)
    tensor = tensor.contiguous()
    tensor = tensor.view(B, C, H, W)
    # tensor = tensor.reshape(B, C, H, W)
    return tensor

# def macro_view(tensor, grid_size=5, macro=3):
#     """[B * grid_size * grid_size, C, patch_size, patch_size]
#     => [B * (grid_size - 2) * (grid_size - 2), C, patch_size*macro, patch_size*macro]"""
#     B, C, patch_size, _ = tensor.size()
#     H = W = patch_size * grid_size
#     B = B / (grid_size ** 2)
#     tensor = tensor.view(B, grid_size, grid_size, C, patch_size, patch_size)
#     tensor = tensor.permute(0, 3, 1, 4, 2, 5)
#     # tensor = tensor.contiguous()
#     tensor = tensor.view(B, C, H, W)
#     return tensor


def make_uid(img_id, dset, sent_idx):
    return "%s_%s_%03d" % (img_id, dset, sent_idx),


class COCODataset(Dataset):
    def __init__(self, split='mscoco_mininval', coco_cap_only=True,
                 loader=default_loader, transform=img_transform):

        self.loader = loader
        self.transform = transform

        coco_img_dir = Path(
            '/home/jaeminc/workspace/datasets/COCO/images/').resolve()
        if split == 'mscoco_train':
            coco_img_dir = coco_img_dir.joinpath('train2014')
        elif split == 'mscoco_minival':
            coco_img_dir = coco_img_dir.joinpath('val2014')
        elif split == 'mscoco_nominival':
            coco_img_dir = coco_img_dir.joinpath('val2014')
        print('# images:', len(list(coco_img_dir.iterdir())))

        data_info_path = project_dir.joinpath(f'data/lxmert/{split}.json')
        with open(data_info_path) as f:
            data_info_dicts = json.load(f)

        print(f"Load {len(data_info_dicts)} data from", data_info_path)

        # flatten
        data = []
        for info_dict in data_info_dicts:
            for dataset, sents in info_dict['sentf'].items():
                if coco_cap_only and dataset != 'mscoco':
                    continue

                for sent_id, sent in enumerate(sents):
                    img_id = info_dict['img_id']
                    img_path = None
                    if coco_img_dir.joinpath(img_id + '.jpg').is_file():
                        img_path = coco_img_dir.joinpath(img_id + '.jpg')
                    elif coco_img_dir.joinpath(img_id + '.png').is_file():
                        img_path = coco_img_dir.joinpath(img_id + '.png')
                    assert img_path is not None

                    datum = {
                        'uid': make_uid(img_id, dataset, sent_id),
                        'img_id': img_id,
                        'img_path': img_path,
                        'sent': sent
                    }
                    data.append(datum)

        self.data = data
        print("# all sentences:", len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        datum = self.data[idx]
        img_path = datum['img_path']
        img = self.loader(img_path)
        img = self.transform(img)
        sent = datum['sent']
        sent = sent_transform(sent)
        return img, sent


class CCDataset(Dataset):
    def __init__(self, split='CC_train',
                 loader=default_loader, transform=img_transform):

        self.loader = loader
        self.transform = transform

        CC_dir = Path('/home/jaeminc/workspace/datasets/CC/').resolve()
        CC_images_dir = CC_dir.joinpath('images')
        if 'train' in split:
            image_dir = CC_images_dir.joinpath('train')
            id2info_path = CC_dir.joinpath('train_id2info.jsonl')
        elif 'valid' in split:
            image_dir = CC_images_dir.joinpath('valid')
            id2info_path = CC_dir.joinpath('valid_id2info.jsonl')
        print('# images:', len(list(image_dir.iterdir())))

        data_info_dicts = []
        with jsonlines.open(id2info_path) as f:
            for i, obj in enumerate(f):
                data_info_dicts.append(obj)

        image_path_list = list(image_dir.iterdir())
        assert len(image_path_list) == len(data_info_dicts)
        self.n_data = len(image_path_list)

        data = []
        for info_dict in data_info_dicts:
            id = info_dict['idx']
            path = image_dir.joinpath(str(id))
            datum = {
                'id': id,
                'img_path': path,
                'sent': info_dict['caption']
            }
            data.append(datum)
        self.data = data
        print("# all data:", len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        datum = self.data[idx]
        img_path = datum['img_path']
        img = self.loader(img_path)
        img = self.transform(img)
        sent = datum['sent']
        # sent = sent_transform(sent)
        return img, sent


def get_loader(split='mscoco_train', mode='train',
               batch_size=32, workers=4, distributed=True, world_size=-1, local_rank=0):

    if 'mscoco' in split:
        dataset = COCODataset(split)
    elif 'CC' in split:
        dataset = CCDataset(split)

    if distributed:
        # sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank)
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


def collate_fn(batch):
    imgs = [img[0] for img in batch]
    sents = [sent[1] for sent in batch]
    B = len(batch)
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    img_batch = torch.zeros((B, 3, h, w), dtype=torch.uint8)
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        if nump_array.ndim < 3:
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.moveaxis(nump_array, -1, 0)  # [H, W, C] => [C, H, W]

        img_batch[i] += torch.from_numpy(nump_array)

    input_ids = torch.LongTensor([f[0] for f in sents])
    input_mask = torch.LongTensor([f[1] for f in sents])
    segment_ids = torch.LongTensor([f[2] for f in sents])
    sents = (input_ids, input_mask, segment_ids)

    return img_batch, sents


class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self._loader = loader
        self.len = len(loader)

        self.stream = torch.cuda.Stream()
        self.mean = torch.tensor(
            [0.485, 0.456, 0.406]).cuda().view(1, 3, 1, 1) * 255
        self.std = torch.tensor(
            [0.229, 0.224, 0.225]).cuda().view(1, 3, 1, 1) * 255
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def __len__(self):
        return self.len

    def preload(self):
        try:
            self.next_img, self.next_sent = next(self.loader)
        except StopIteration:
            self.next_img = None
            self.next_sent = None
            return
        # if record_stream() doesn't work, another option is to make sure device imgs are created
        # on the main stream.
        # self.next_img_gpu = torch.empty_like(self.next_img, device='cuda')
        # self.next_sent_gpu = torch.empty_like(self.next_sent, device='cuda')
        # Need to make sure the memory allocated for next_* is not still in use by the main stream
        # at the time we start copying to next_*:
        # self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self.next_img = self.next_img.cuda(non_blocking=True)
            self.next_sent = (self.next_sent[0].cuda(non_blocking=True),
                              self.next_sent[1].cuda(non_blocking=True),
                              self.next_sent[2].cuda(non_blocking=True))
            # more code for the alternative if record_stream() doesn't work:
            # copy_ will record the use of the pinned source tensor in this side stream.
            # self.next_img_gpu.copy_(self.next_img, non_blocking=True)
            # self.next_sent_gpu.copy_(self.next_sent, non_blocking=True)
            # self.next_img = self.next_img_gpu
            # self.next_sent = self.next_sent_gpu

            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_img = self.next_img.half()
            # else:
            self.next_img = self.next_img.float()
            self.next_img = self.next_img.sub_(self.mean).div_(self.std)

            # self.next_img = grid_view(self.next_img)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        img = self.next_img
        sent = self.next_sent
        if img is not None:
            img.record_stream(torch.cuda.current_stream())
        if sent is not None:
            # sent.record_stream(torch.cuda.current_stream())
            sent[0].record_stream(torch.cuda.current_stream())
            sent[1].record_stream(torch.cuda.current_stream())
            sent[2].record_stream(torch.cuda.current_stream())
        self.preload()
        return img, sent


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
