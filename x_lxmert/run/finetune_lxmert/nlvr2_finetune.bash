# The name of this experiment.
name=lxmert_nlvr

# Save logs and models under snap/nlvr2; Make backup.
output=snap/nlvr2/$name
mkdir -p $output/src
cp -r src/* $output/src/
cp $0 $output/run.bash

# See run/Readme.md for option details.
CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PYTHONPATH:./src \
    python src/tasks/nlvr2.py \
    --train train --valid valid \
    --llayers 9 --xlayers 5 --rlayers 5 \
    --lr 5e-5 \
    --numWorkers 8 \
    --tqdm --output $output ${@:2} \
    --encoder maskrcnn \
    --feed_exact_feat \
    --n_boxes 36 \
    --im_ratio original \
    --loadLXMERT snap/pretrain/lxmert_reimplementation_fp16_noaccumulation_fulldata/Epoch12
    # --loadLXMERT snap/pretrain/grid_cluster_lxrt_COCOVG_W15_VMP_Match_vocab10000_resnext152_maskrcnn_dim2048_vis_sampling_fp16_noaccumulation_fulldata/Epoch12 \
    # --loadLXMERT snap/pretrain/lxmert_reimplementation/Epoch20 \
    # --batchSize 320 \
    # --optim bert \
    # --load snap/nlvr2/vgbert_nlvr/BEST


# --loadLXMERT snap/pretrain/vgbert/BEST_EVAL_LOSS_LXRT \
