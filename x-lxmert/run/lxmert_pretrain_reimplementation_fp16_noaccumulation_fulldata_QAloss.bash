# The name of experiment
name=lxmert_reimplementation_fp16_noaccumulation_fulldata_QAloss

# Create dirs and make backup
output=snap/pretrain/$name
mkdir -p $output/src
cp -r src/* $output/src/
cp $0 $output/run.bash

# Pre-training
CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PYTHONPATH:./src \
    python src/pretrain/lxmert_pretrain.py \
    --taskMaskLM --taskObjPredict --taskMatched --taskQA \
    --visualLosses obj,feat \
    --wordMaskRate 0.15 --objMaskRate 0.15 \
    --train mscoco_train,mscoco_nominival,vgnococo \
    --valid mscoco_minival \
    --llayers 9 --xlayers 5 --rlayers 5 \
    --optim bert --lr 1e-4 \
    --epochs 20 \
    --tqdm --output $output ${@:2} \
    --numWorkers 4 \
    --encoder maskrcnn \
    --n_class 1600 \
    --distributed --multiGPU --mixed_precision \
    --feat_dim 2048 \
    --im_ratio original \
    --feed_exact_feat --target_exact_feat --target_obj_id \
    --n_boxes 36 \
    --update 1 \
    # --train mscoco_train,mscoco_nominival,vgnococo \
    # --load snap/pretrain/lxmert_reimplementation/Epoch12
    # --train mscoco_train,mscoco_nominival,vgnococo --valid mscoco_minival \
