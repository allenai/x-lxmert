# The name of this experiment.
# name=$2

# # Save logs and models under snap/nlvr2; make backup.
# output=snap/nlvr2/$name
# mkdir -p $output/src
# cp -r src/* $output/src/
# cp $0 $output/run.bash

# See Readme.md for option details.
CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PYTHONPATH:./src \
    python src/tasks/nlvr2.py \
    --test \
    --grid_model \
    --backbone lxmert \
    --llayers 9 --xlayers 5 --rlayers 5 \
    --grid_size 8 \
    --encoder maskrcnn --im_ratio original \
    --feat_dim 2048 \
    --numWorkers 6 \
    --load snap/nlvr2/grid8_lxmert_QAloss_nlvr_adamw_clip5/BEST \
    --comment 'grid8_lxmert_QAloss'
