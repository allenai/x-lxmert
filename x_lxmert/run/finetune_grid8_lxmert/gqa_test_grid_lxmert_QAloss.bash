# The name of this experiment.
# name=lxmert_gqa_test

# # Save logs and models under snap/vqa; make backup.
# output=snap/gqa/$name
# mkdir -p $output/src
# cp -r src/* $output/src/
# cp $0 $output/run.bash

# See Readme.md for option details.
CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PYTHONPATH:./src \
    python src/tasks/gqa.py \
    --test \
    --llayers 9 --xlayers 5 --rlayers 5 \
    --grid_model \
    --backbone lxmert \
    --grid_size 8 \
    --tqdm \
    --encoder maskrcnn --im_ratio original \
    --feed_exact_feat --feat_dim 2048 \
    --numWorkers 4 \
    --load snap/gqa/grid8_lxmert_QAloss_gqa/BEST \
    --comment 'grid_lxmert_QAloss' ${@:2} \
