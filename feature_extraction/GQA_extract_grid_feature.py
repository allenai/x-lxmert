# !/usr/bin/env python


from PIL import Image

import argparse
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms, models
from torchvision.datasets.folder import default_loader
import h5py
import numpy as np
import cv2


if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.model_serialization import load_state_dict
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.image_list import to_image_list

def build_mask_rcnn():
    ckpt_dir = Path('./mask_rcnn_ckpt').resolve()
    config_path = ckpt_dir.joinpath('detectron_config.yaml')
    model_path = ckpt_dir.joinpath('detectron_model.pth')

    cfg.merge_from_file(str(config_path))
    cfg.freeze()
    model = build_detection_model(cfg)
    checkpoint = torch.load(str(model_path), map_location=torch.device("cpu"))

    load_state_dict(model, checkpoint.pop("model"))

    return model


def get_grid_bbox(grid_size=8, image_size=(448,448)):
    H, W = image_size
    grid_h = H // grid_size
    grid_w = W // grid_size

    bbox = np.zeros(shape=(grid_size, grid_size, 4), dtype=np.float32)

    for i in range(grid_size):
        for j in range(grid_size):
            # pre-normalize (0 ~ 1)
            x0, x1 = j * grid_w, (j + 1) * grid_w
            y0, y1 = i * grid_h, (i + 1) * grid_h
            coordinate = (x0, y0, x1, y1)
            bbox[i, j] = coordinate
    return bbox


MAX_SIZE = 1333
MIN_SIZE = 800

def image_transform(path):

    img = Image.open(path)
    im = np.array(img).astype(np.float32)
    # IndexError: too many indices for array, grayscale images
    if len(im.shape) < 3:
        im = np.repeat(im[:, :, np.newaxis], 3, axis=2)
    im = im[:, :, ::-1]
    im -= np.array([102.9801, 115.9465, 122.7717])
    im_shape = im.shape
    im_height = im_shape[0]
    im_width = im_shape[1]
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    if args.imsize:
        im = cv2.resize(im, (args.imsize, args.imsize), interpolation=cv2.INTER_LINEAR)

    else:
        # Scale based on minimum size
        im_scale = MIN_SIZE / im_size_min

        # Prevent the biggest axis from being more than max_size
        # If bigger, scale it down
        if np.round(im_scale * im_size_max) > MAX_SIZE:
            im_scale = MAX_SIZE / im_size_max

        x_resize = int(im_scale * im_width)
        x_resize = x_resize - (x_resize % 32)

        y_resize = int(im_scale * im_height)
        y_resize = y_resize - (y_resize % 32)

        im = cv2.resize(im, (x_resize, y_resize), interpolation=cv2.INTER_LINEAR)


    # im = cv2.resize(
    #     im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR
    # )
    img_tensor = torch.from_numpy(im).permute(2, 0, 1)

    # im_info = {"width": im_width, "height": im_height}

    # return img, im_scale, im_info
    return img_tensor


class GQADataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        print('Load images from', image_dir)
        total_path_list = list(image_dir.rglob('*.jpg'))
        print('total # image:', len(total_path_list))

        self.image_path_list = []
        for image_path in tqdm(total_path_list):
            try:
                with Image.open(image_path) as im:
                    pass
                self.image_path_list.append(image_path)
            except IOError:
                continue
        self.n_images = len(self.image_path_list)
        print('Normal # image:', self.n_images)

        self.transform = image_transform

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        image_path = self.image_path_list[idx]
        image_id = image_path.stem

        # img_tensor, im_scale, im_info = self.transform(image_path)
        img_tensor = self.transform(image_path)

        _, H, W = img_tensor.size()
        bbox = get_grid_bbox(grid_size=args.grid_size, image_size=(H, W))
        bbox = torch.from_numpy(bbox.reshape(args.grid_size**2, 4))
        boxlist = BoxList(bbox, image_size=(W, H))


        return {
            'img_tensor': img_tensor,
            # 'img_scale': im_scale,
            # 'img_info': im_info,
            'img_id': image_id,
            'boxlist': boxlist,
        }


def collate_fn(batch):
    B = len(batch)

    # img_tensor = torch.zeros((B, 3, args.img_size, args.img_size), dtype=torch.float)
    img_tensors = []
    img_ids = []
    # img_scales = []
    # img_infos = []

    grid_proposals = []

    for i, entry in enumerate(batch):
        img_tensors.append(entry['img_tensor'])

        img_ids.append(entry['img_id'])
        # img_scales.append(entry['img_scale'])
        # img_infos.append(entry['img_info'])

        grid_proposals.append(entry['boxlist'])

    batch_out = {}
    batch_out['img_tensor'] = torch.stack(img_tensors).float()
    batch_out['img_id'] = img_ids
    # batch_out['img_scale'] = img_scales
    # batch_out['img_infos'] = img_infos

    batch_out['grid_proposals'] = grid_proposals

    return batch_out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract grid features using mask r-cnn')
    parser.add_argument('--gqaroot', type=str, default='../datasets/GQA/')
    # parser.add_argument('--split', type=str, default='valid')
    parser.add_argument('--grid_size', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--backbone', type=str, default='maskrcnn')
    parser.add_argument('--im_ratio', type=str, default='original')
    parser.add_argument('--imsize', type=int, default=None)

    args = parser.parse_args()

    assert args.backbone == 'maskrcnn'
    assert args.batch_size == 1
    model = build_mask_rcnn()
    model.to(device)
    model.eval()
    dim = 2048

    print(args)
    if args.imsize:
        print(f'Images will be resize to {(args.imsize, args.imsize)} and encoded to {(dim, args.grid_size, args.grid_size)} features')
    else:
        print(f'Images will be encoded to {(dim, args.grid_size, args.grid_size)} features')

    gqa_dir = Path(args.gqaroot).resolve()
    gqa_img_dir = gqa_dir.joinpath('images')

    dataset = GQADataset(gqa_img_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=False, collate_fn=collate_fn, num_workers=4)

    out_dir = gqa_dir.joinpath('features').resolve()
    if not out_dir.exists():
        out_dir.mkdir()
    if args.imsize:
        output_fname = out_dir.joinpath(
            f'{args.backbone}_grid{args.grid_size}_imsize{args.imsize}.h5')
    else:
        output_fname = out_dir.joinpath(
            f'{args.backbone}_grid{args.grid_size}.h5')
    print('Output features will be saved at', output_fname)

    with h5py.File(output_fname, 'w') as f:
        with torch.no_grad():

            for i, batch in tqdm(enumerate(dataloader),
                                 desc=f'{args.backbone}_{(dim, args.grid_size, args.grid_size)}', ncols=150,
                                 total=len(dataloader)):

                x = batch['img_tensor'].to(device)
                img_ids = batch['img_id']
                grid_proposals = batch['grid_proposals']

                for boxlist in grid_proposals:
                    boxlist.bbox = boxlist.bbox.to(device)

                B, C, H, W = x.size()
                assert B == 1

                fpn_features = model.backbone(x)
                out, _, _ = model.roi_heads(fpn_features, grid_proposals)
                feature = out['fc6']

                assert feature.size() == (B * args.grid_size * args.grid_size, dim), feature.size()

                feature = feature.cpu().detach().view(B, args.grid_size,  args.grid_size, dim).numpy()  # [B, 14, 14, 2048]

                for i, img_id in enumerate(img_ids):
                    grp = f.create_group(img_id)
                    grp['features'] = feature[i]
