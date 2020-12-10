# The name of experiment
name=x-lxmert

# Create dirs and make backup
output=snap/pretrain/$name
mkdir -p $output/src
cp -r src/* $output/src/
cp $0 $output/run.bash

# Pre-training
CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PYTHONPATH:./src \
    python src/pretrain/lxmert_pretrain.py \
    --grid_model \
    --taskMaskLM --taskObjPredict --taskMatched \
    --visualLosses obj \
    --vis_mask_predict \
    --wordMaskRate 0.15 \
    --train mscoco_train,mscoco_nominival,vgnococo --valid mscoco_minival \
    --llayers 9 --xlayers 5 --rlayers 5 \
    --optim bert --lr 1e-4 --epochs 20 \
    --warmup_ratio --0.05 \
    --tqdm --output $output ${@:2} \
    --numWorkers 3 \
    --clustering \
    --encoder maskrcnn \
    --im_ratio original \
    --grid_size 8 --feat_dim 2048 \
    --distributed --multiGPU --mixed_precision \
    --update 1 \
    --clip_grad_norm 1.0 \
    --vis_mask_COCOVG_only