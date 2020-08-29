# The name of experiment
name=grid8_lxmert_COCOVG_W15_V15_feat_Match_vocab10000_maskrcnn_dim2048_fp16_fulldata

# Create dirs and make backup
output=snap/pretrain/$name
mkdir -p $output/src
cp -r src/* $output/src/
cp $0 $output/run.bash

# Pre-training
CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PYTHONPATH:./src \
    python src/pretrain/lxmert_pretrain.py \
    --taskMaskLM --taskObjPredict --taskMatched \
    --visualLosses feat \
    --wordMaskRate 0.15 \
    --objMaskRate 0.15 \
    --train mscoco_train,mscoco_nominival,vgnococo --valid mscoco_minival \
    --llayers 9 --xlayers 5 --rlayers 5 \
    --optim bert --lr 1e-4 --epochs 20 \
    --tqdm --output $output ${@:2} \
    --numWorkers 2 \
    --distributed --multiGPU --mixed_precision \
    --update 1 \
    --clip_grad_norm 1.0 \
    --encoder maskrcnn \
    --grid_model --grid_size 8 \
    --im_ratio original \
    --feed_exact_feat \
    --target_exact_feat \
    # --target_obj_id \
    # --target_prob \
    # --imsize 1120
    # --loadLXMERT snap/pretrain/grid_lxmert/Epoch12 \
    # --fromScratch \
    # --batchSize 560
    # --train mscoco_train,mscoco_nominival,vgnococo --valid mscoco_minival \
    # --train mscoco_train,mscoco_nominival,vgnococo --valid mscoco_minival \
    # --train mscoco_minival  --valid mscoco_minival \
