We used the same version of mask r-cnn used in [vilbert multi-task](https://github.com/facebookresearch/vilbert-multi-task/tree/master/data).

```
wget -O model_final.pth https://dl.fbaipublicfiles.com/vilbert-multi-task/detectron_model.pth
wget -O e2e_faster_rcnn_X-152-32x8d-FPN_1x_MLP_2048_FPN_512_train.yaml https://dl.fbaipublicfiles.com/vilbert-multi-task/detectron_config.yaml
```
