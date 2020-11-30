
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

# Extract features

```bash
# Grid features
python coco_extract_grid_feature.py
python VG_extract_grid_feature.py
python GQA_extract_grid_feature.py
python nlvr2_extract_grid_feature.py

# bounding box features (optional)
python coco_extract_bbox_feature.py
python VG_extract_bbox_feature.py
python GQA_extract_bbox_feature.py
python nlvr2_extract_bbox_feature.py
```

# K-means clustering

We use [faiss](https://github.com/facebookresearch/faiss) implementaion of K-means clustering.

## Install

Please follow [the original installation guide](https://github.com/facebookresearch/faiss/blob/master/INSTALL.md).

## Run K-means clustering

```bash

python run_kmeans.py
```