# The name of this experiment.
name=grid8_cluster_lxmert_QAloss_vqa

# Save logs and models under snap/vqa; make backup.
output=snap/vqa/$name
mkdir -p $output/src
cp -r src/* $output/src/
cp $0 $output/run.bash

# See Readme.md for option details.
CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PYTHONPATH:./src \
    python src/tasks/vqa.py \
    --train train,nominival --valid minival \
    --grid_model \
    --backbone lxmert \
    --llayers 9 --xlayers 5 --rlayers 5 \
    --optim bert \
    --lr 5e-4 --epochs 10 \
    --tqdm --output $output ${@:2} \
    --grid_size 8 \
    --encoder maskrcnn --im_ratio original \
    --feat_dim 2048 \
    --numWorkers 6 \
    --loadLXMERTQA snap/pretrain/grid8_cluster_lxrt_COCOVG_W15_VMP_Match_vocab10000_maskrcnn_dim2048_fp16_QAloss/Epoch20
    # --loadLXMERT snap/pretrain/lxmert_reimplementation_fp16_noaccumulation_fulldata/Epoch12
    # --loadLXMERT snap/pretrain/lxmert_reimplementation_fullp_noaccumulation/Epoch12
    # --loadLXMERT snap/pretrain/lxmert_reimplementation/Epoch20 \
