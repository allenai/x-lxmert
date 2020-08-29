# The name of this experiment.
name=lxmert_QAloss_vqa

# Save logs and models under snap/vqa; make backup.
output=snap/vqa/$name
mkdir -p $output/src
cp -r src/* $output/src/
cp $0 $output/run.bash

# See Readme.md for option details.
CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PYTHONPATH:./src \
    python src/tasks/vqa.py \
    --train train,nominival --valid minival  \
    --llayers 9 --xlayers 5 --rlayers 5 \
    --optim bert \
    --lr 5e-4 --epochs 10 \
    --tqdm --output $output ${@:2} \
    --numWorkers 8 \
    --encoder maskrcnn \
    --feed_exact_feat \
    --n_boxes 36 \
    --im_ratio original \
    --loadLXMERTQA snap/pretrain/lxmert_reimplementation_fp16_noaccumulation_fulldata_QAloss/Epoch20
    # --loadLXMERT snap/pretrain/lxmert_reimplementation_fullp_noaccumulation/Epoch12
    # --loadLXMERT snap/pretrain/lxmert_reimplementation/Epoch20 \
