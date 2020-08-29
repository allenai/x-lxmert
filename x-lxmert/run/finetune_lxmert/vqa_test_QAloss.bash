# The name of this experiment.
# name=lxmert_vqa_test

# # Save logs and models under snap/vqa; make backup.
# output=snap/vqa/$name
# mkdir -p $output/src
# cp -r src/* $output/src/
# cp $0 $output/run.bash

# See Readme.md for option details.
CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PYTHONPATH:./src \
    python src/tasks/vqa.py \
    --test \
    --llayers 9 --xlayers 5 --rlayers 5 \
    --tqdm \
    --encoder maskrcnn --im_ratio original \
    --feed_exact_feat --feat_dim 2048 \
    --n_boxes 36 \
    --numWorkers 8 \
    --load snap/vqa/lxmert_QAloss_vqa/BEST \
    --comment 'lxmert_QAloss'
    # --load snap/vqa/lxmert_vqa/BEST \
