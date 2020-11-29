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
from types import SimpleNamespace


# SPLIT to its folder name under IMG_ROOT
SPLIT2DIR = {
    'train': 'train',
    'valid': 'dev',
    'test': 'test1',
}
# SPLIT2DIR = {
#     'train': 'train2017',
#     'valid': 'val2017',
#     'test': 'test2017',
# }

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'



from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.layers import nms
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.model_serialization import load_state_dict
from maskrcnn_benchmark.structures.image_list import to_image_list

NUM_FEATURES = 36
MAX_SIZE = 1333
MIN_SIZE = 800

class FeatureExtractor:
    def __init__(self):
        self.args = self.get_parser()
        self.args.num_features = NUM_FEATURES
        self.detection_model = self._build_detection_model()

    def get_parser(self):
        ckpt_dir = Path('./mask_rcnn_ckpt').resolve()
        config_path = ckpt_dir.joinpath('detectron_config.yaml')
        model_path = ckpt_dir.joinpath('detectron_model.pth')

        parser = SimpleNamespace(model_file=str(model_path),
                                config_file=str(config_path),
                                batch_size=1,
                                num_features=100,
                                feature_name="fc6",
                                confidence_threshold=0,
                                background=False,
                                partition=0)
        return parser

    def _build_detection_model(self):
        cfg.merge_from_file(self.args.config_file)
        cfg.freeze()

        model = build_detection_model(cfg)
        checkpoint = torch.load(self.args.model_file, map_location=torch.device("cpu"))

        load_state_dict(model, checkpoint.pop("model"))

        model.to("cuda")
        model.eval()
        return model

    def _process_feature_extraction(self, output, im_scales, im_infos, feature_name="fc6", conf_thresh=0):
        batch_size = len(output[0]["proposals"])
        n_boxes_per_image = [len(boxes) for boxes in output[0]["proposals"]]
        score_list = output[0]["scores"].split(n_boxes_per_image)
        score_list = [torch.nn.functional.softmax(x, -1) for x in score_list]
        feats = output[0][feature_name].split(n_boxes_per_image)
        cur_device = score_list[0].device

        feat_list = []
        info_list = []

        for i in range(batch_size):
            dets = output[0]["proposals"][i].bbox / im_scales[i]
            scores = score_list[i]
            max_conf = torch.zeros((scores.shape[0])).to(cur_device)
            conf_thresh_tensor = torch.full_like(max_conf, conf_thresh)
            start_index = 1
            # Column 0 of the scores matrix is for the background class
            if self.args.background:
                start_index = 0
            for cls_ind in range(start_index, scores.shape[1]):
                cls_scores = scores[:, cls_ind]
                keep = nms(dets, cls_scores, 0.5)
                max_conf[keep] = torch.where(
                    # Better than max one till now and minimally greater than conf_thresh
                    (cls_scores[keep] > max_conf[keep])
                    & (cls_scores[keep] > conf_thresh_tensor[keep]),
                    cls_scores[keep],
                    max_conf[keep],
                )

            sorted_scores, sorted_indices = torch.sort(max_conf, descending=True)
            num_boxes = (sorted_scores[: self.args.num_features] != 0).sum()
            keep_boxes = sorted_indices[: self.args.num_features]
            feat_list.append(feats[i][keep_boxes])
            bbox = output[0]["proposals"][i][keep_boxes].bbox / im_scales[i]
            # Predict the class label using the scores
            # objects = torch.argmax(scores[keep_boxes, start_index:], dim=1)
            cls_prob, objects = torch.max(scores[keep_boxes, start_index:], dim=1)
            info_list.append(
                {
                    "bbox": bbox.cpu().numpy(),
                    "num_boxes": num_boxes.item(),
                    "objects": objects.cpu().numpy(),
                    "image_width": im_infos[i]["width"],
                    "image_height": im_infos[i]["height"],
                    #"cls_prob": scores[keep_boxes].cpu().numpy() # or cls_prob.cpu().numpy() (if background probability is not needed)
                }
            )
        return feat_list, info_list

    def get_detectron_features(self, batch):
        # img_tensor, im_scales, im_infos = [], [], []
        #
        # for image_path in image_paths:
        #     im, im_scale, im_info = self._image_transform(image_path)
        #     img_tensor.append(im)
        #     im_scales.append(im_scale)
        #     im_infos.append(im_info)

        img_tensor = batch['img_tensor']
        im_scales = batch['im_scales']
        im_infos = batch['im_infos']

        # Image dimensions should be divisible by 32, to allow convolutions
        # in detector to work
        current_img_list = to_image_list(img_tensor, size_divisible=32)
        current_img_list = current_img_list.to("cuda")

        with torch.no_grad():
            output = self.detection_model(current_img_list)

        feat_list = self._process_feature_extraction(
            output,
            im_scales,
            im_infos,
            self.args.feature_name,
            self.args.confidence_threshold,
        )

        return feat_list

    def _chunks(self, array, chunk_size):
        for i in range(0, len(array), chunk_size):
            yield array[i : i + chunk_size]

    def _save_feature(self, file_name, feature, info):
        file_base_name = os.path.basename(file_name)
        file_base_name = file_base_name.split(".")[0]
        info["image_id"] = file_base_name
        info["features"] = feature.cpu().numpy()
        file_base_name = file_base_name + ".npy"

        np.save(os.path.join(self.args.output_folder, file_base_name), info)

    def extract_features(self, image_path):

        features, infos = self.get_detectron_features([image_path])

        return features, infos


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

    # Scale based on minimum size
    im_scale = MIN_SIZE / im_size_min

    # Prevent the biggest axis from being more than max_size
    # If bigger, scale it down
    if np.round(im_scale * im_size_max) > MAX_SIZE:
        im_scale = MAX_SIZE / im_size_max

    im = cv2.resize(
        im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR
    )
    img = torch.from_numpy(im).permute(2, 0, 1)

    im_info = {"width": im_width, "height": im_height}

    return img, im_scale, im_info


class NLVR2Dataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.image_path_list = list(tqdm(image_dir.rglob('*.png')))
        self.n_images = len(self.image_path_list)

        self.transform = image_transform

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        image_path = self.image_path_list[idx]
        image_id = image_path.stem

        img_tensor, im_scale, im_info = self.transform(image_path)

        return {
            'img_id': image_id,
            'img_tensor': img_tensor,
            'im_scale': im_scale,
            'im_info': im_info,
        }


def collate_fn(batch):
    img_ids = []
    img_tensors = []
    img_scales = []
    img_infos = []

    for i, entry in enumerate(batch):
        img_ids.append(entry['img_id'])
        img_tensors.append(entry['img_tensor'])
        img_scales.append(entry['im_scale'])
        img_infos.append(entry['im_info'])

    batch_out = {}
    batch_out['img_ids'] = img_ids
    batch_out['img_tensor'] = img_tensors
    batch_out['im_scales'] = img_scales
    batch_out['im_infos'] = img_infos

    return batch_out



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract bounding box features using mask r-cnn')
    parser.add_argument('--nlvrroot', type=str, default='../datasets/nlvr2/')
    parser.add_argument('--split', type=str, default='valid')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--backbone', type=str, default='maskrcnn')

    args = parser.parse_args()

    assert args.backbone == 'maskrcnn'
    feature_extractor = FeatureExtractor()
    dim = 2048

    print(args)
    print(f'{NUM_FEATURES} boxes and features ({dim}) will be extracted')

    nlvr_dir = Path(args.nlvrroot).resolve()
    nlvr_img_dir = nlvr_dir.joinpath('images')
    nlvr_img_split_dir = nlvr_img_dir.joinpath(SPLIT2DIR[args.split])
    print('Load images from', nlvr_img_split_dir)

    dataset = NLVR2Dataset(nlvr_img_split_dir)
    print('# Images:', len(dataset))
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=False, collate_fn=collate_fn, num_workers=4)

    out_dir = nlvr_dir.joinpath('features')
    if not out_dir.exists():
        out_dir.mkdir()
    output_fname = out_dir.joinpath(f'{args.backbone}_{args.split}_boxes{NUM_FEATURES}.h5')
    print('Output features will be saved at', output_fname)

    with h5py.File(output_fname, 'w') as f:
        with torch.no_grad():
            for i, batch in tqdm(enumerate(dataloader),
                                 desc=f'{args.backbone}_{(NUM_FEATURES, dim)}_{args.split}', ncols=150,
                                 total=len(dataloader)):

                img_ids = batch['img_ids']
                feat_list, info_list = feature_extractor.get_detectron_features(batch)

                for img_id, feat, info in zip(img_ids, feat_list, info_list):

                    assert info['num_boxes'] == NUM_FEATURES

                    assert len(feat) == NUM_FEATURES, ('feat', feat)
                    assert len(info['objects']) == NUM_FEATURES, ('obj_ids', info['objects'].shape)
                    assert len(info['bbox']) == NUM_FEATURES, ('bbox', info['bbox'].shape)

                    grp = f.create_group(img_id)
                    grp['features'] = feat.cpu().detach().numpy() # [num_features, 2048]
                    grp['obj_id'] = info['objects']
                    grp['boxes'] = info['bbox']
                    grp['img_w'] = info['image_width']
                    grp['img_h'] = info['image_height']
