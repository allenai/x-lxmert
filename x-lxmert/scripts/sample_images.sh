PYTHONPATH=$PYTHONPATH:./src \
PYTHONPATH=$PYTHONPATH:../image_generator/src \
python src/sample_images.py \
    --numWorkers 0 \
    --backbone "lxmert" \
    --grid_model \
    --grid_size 8 \
    --resize_target_size 256 \
    --clustering \
    --feat_dim 2048 \
    --codebook_dim 2048 \
    --n_codebook 10000 \
    --norm_type 'spade_in'
    --vis_mask_predict \
    --sample_steps 4 \
    --load "snap/pretrained/x-lxmert/Epoch20" \
