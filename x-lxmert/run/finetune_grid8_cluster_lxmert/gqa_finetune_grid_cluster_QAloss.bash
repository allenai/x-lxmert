# The name of this experiment.
name=grid8_cluster_lxmert_QAloss_cocovismask_gqa

# Save logs and models under snap/gqa; make backup.
output=snap/gqa/$name
mkdir -p $output/src
cp -r src/* $output/src/
cp $0 $output/run.bash

# See Readme.md for option details.
CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PYTHONPATH:./src \
    python src/tasks/gqa.py \
    --train train,valid --valid testdev \
    --grid_model \
    --backbone lxmert \
    --llayers 9 --xlayers 5 --rlayers 5 \
    --lr 1e-5 \
    --optim bert \
    --tqdm --output $output ${@:2} \
    --grid_size 8 \
    --encoder maskrcnn --im_ratio original \
    --feat_dim 2048 \
    --numWorkers 6 \
    --loadLXMERTQA snap/pretrain/grid8_cluster_lxrt_COCOVG_W15_VMP_Match_vocab10000_maskrcnn_dim2048_fp16_cocovismask_QAloss/Epoch20
    # --load snap/gqa/lxmert_gqa/Epoch80
    # --loadLXMERT snap/pretrain/lxmert_reimplementation_fp16_noaccumulation_fulldata/Epoch12

