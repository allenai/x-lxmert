
# Feature extraction

We use the same version of 'Bottom-up and Top-down feature extractor' used in [vilbert multi-task](https://github.com/facebookresearch/vilbert-multi-task/tree/master/data).
This is based on [Vedanuj Goswami's mask-rcnn-benchmark fork](https://gitlab.com/vedanuj/vqa-maskrcnn-benchmark).


## Install

Please follow [the original installation guide](https://gitlab.com/vedanuj/vqa-maskrcnn-benchmark/-/blob/master/INSTALL.md).

## Download checkpoint

```
wget -O model_final.pth https://dl.fbaipublicfiles.com/vilbert-multi-task/detectron_model.pth
wget -O e2e_faster_rcnn_X-152-32x8d-FPN_1x_MLP_2048_FPN_512_train.yaml https://dl.fbaipublicfiles.com/vilbert-multi-task/detectron_config.yaml
```

# Download or extract features

## Download features

```bash

# Grid features
wget https://ai2-vision-x-lxmert.s3-us-west-2.amazonaws.com/butd_features/COCO/maskrcnn_train_grid8.h5 -P datasets/COCO/features
wget https://ai2-vision-x-lxmert.s3-us-west-2.amazonaws.com/butd_features/COCO/maskrcnn_valid_grid8.h5 -P datasets/COCO/features
wget https://ai2-vision-x-lxmert.s3-us-west-2.amazonaws.com/butd_features/COCO/maskrcnn_test_grid8.h5 -P datasets/COCO/features

wget https://ai2-vision-x-lxmert.s3-us-west-2.amazonaws.com/butd_features/VG/maskrcnn_grid8.h5 -P datasets/VG/features

wget https://ai2-vision-x-lxmert.s3-us-west-2.amazonaws.com/butd_features/GQA/maskrcnn_grid8.h5 -P datasets/GQA/features

wget https://ai2-vision-x-lxmert.s3-us-west-2.amazonaws.com/butd_features/NLVR2/maskrcnn_train_grid8.h5 -P datasets/NLVR2/features
wget https://ai2-vision-x-lxmert.s3-us-west-2.amazonaws.com/butd_features/NLVR2/maskrcnn_valid_grid8.h5 -P datasets/NLVR2/features
wget https://ai2-vision-x-lxmert.s3-us-west-2.amazonaws.com/butd_features/NLVR2/maskrcnn_test_grid8.h5 -P datasets/NLVR2/features

# bounding box features (optional)
wget https://ai2-vision-x-lxmert.s3-us-west-2.amazonaws.com/butd_features/COCO/maskrcnn_train_boxes36.h5 -P datasets/COCO/features
wget https://ai2-vision-x-lxmert.s3-us-west-2.amazonaws.com/butd_features/COCO/maskrcnn_valid_boxes36.h5 -P datasets/COCO/features
wget https://ai2-vision-x-lxmert.s3-us-west-2.amazonaws.com/butd_features/COCO/maskrcnn_test_boxes36.h5 -P datasets/COCO/features


wget https://ai2-vision-x-lxmert.s3-us-west-2.amazonaws.com/butd_features/VG/maskrcnn_boxes36.h5 -P datasets/VG/features

wget https://ai2-vision-x-lxmert.s3-us-west-2.amazonaws.com/butd_features/GQA/maskrcnn_boxes36.h5 -P datasets/GQA/features

wget https://ai2-vision-x-lxmert.s3-us-west-2.amazonaws.com/butd_features/NLVR2/maskrcnn_train_boxes36.h5 -P datasets/NLVR2/features
wget https://ai2-vision-x-lxmert.s3-us-west-2.amazonaws.com/butd_features/NLVR2/maskrcnn_valid_boxes36.h5 -P datasets/NLVR2/features
wget https://ai2-vision-x-lxmert.s3-us-west-2.amazonaws.com/butd_features/NLVR2/maskrcnn_test_boxes36.h5 -P datasets/NLVR2/features
```


## Extract features (optional)

```bash
cd feature_extraction

# Grid features
python coco_extract_grid_feature.py --split train
python coco_extract_grid_feature.py --split valid
python coco_extract_grid_feature.py --split test

python VG_extract_grid_feature.py

python GQA_extract_grid_feature.py

python nlvr2_extract_grid_feature.py --split train
python nlvr2_extract_grid_feature.py --split valid
python nlvr2_extract_grid_feature.py --split test

# bounding box features (optional)
python coco_extract_bbox_feature.py --split train
python coco_extract_bbox_feature.py --split valid
python coco_extract_bbox_feature.py --split test

python VG_extract_bbox_feature.py

python GQA_extract_bbox_feature.py

python nlvr2_extract_bbox_feature.py --split train
python nlvr2_extract_bbox_feature.py --split valid
python nlvr2_extract_bbox_feature.py --split test
```




# K-means clustering

We use [faiss](https://github.com/facebookresearch/faiss) implementaion of K-means clustering.

## Install

Please follow [the original installation guide](https://github.com/facebookresearch/faiss/blob/master/INSTALL.md).

## Run K-means clustering

```bash

python run_kmeans.py --src=mscoco_train --tgt mscoco_train mscoco valid vg
```