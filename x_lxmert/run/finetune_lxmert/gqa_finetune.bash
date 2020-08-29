# The name of this experiment.
name=lxmert_gqa

# Save logs and models under snap/gqa; make backup.
output=snap/gqa/$name
mkdir -p $output/src
cp -r src/* $output/src/
cp $0 $output/run.bash

# See Readme.md for option details.
CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PYTHONPATH:./src \
    python src/tasks/gqa.py \
    --train train,valid --valid testdev \
    --llayers 9 --xlayers 5 --rlayers 5 \
    --lr 1e-5 \
    --optim bert \
    --numWorkers 8 \
    --tqdm --output $output ${@:2} \
    --encoder maskrcnn \
    --feed_exact_feat \
    --n_boxes 36 \
    --im_ratio original \
    --loadLXMERT snap/pretrain/lxmert_reimplementation_fp16_noaccumulation_fulldata/Epoch20
    # --load snap/gqa/lxmert_gqa/Epoch80
    # --loadLXMERT snap/pretrain/lxmert_reimplementation_fp16_noaccumulation_fulldata/Epoch12

