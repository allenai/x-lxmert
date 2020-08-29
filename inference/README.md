# Prerequisites

1) Install python dependencies
2) Download checkpoints

# Command

```
bash sample.sh

or

python sample_images.py \
    --numWorkers 0 \
    --backbone "lxmert" \
    --grid_model \
    --grid_size 8 \
    --resize_target_size 256 \
    --clustering \
    --feat_dim 2048 \
    --codebook_dim 2048 \
    --n_codebook 10000 \
    --encoder "maskrcnn" \
    --im_ratio "original" \
    --vis_mask_predict \
    --sample_steps 4 \
    --load "../x-lxmert/snap/pretrained/x-lxmert/Epoch20" \
```
