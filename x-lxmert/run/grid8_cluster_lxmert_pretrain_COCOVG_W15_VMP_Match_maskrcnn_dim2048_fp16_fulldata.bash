# The name of experiment
name=grid8_cluster_lxrt_COCOVG_W15_VMP_Match_vocab10000_maskrcnn_dim2048_fp16_fulldata

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
    --tqdm --output $output ${@:2} \
    --numWorkers 4 \
    --clustering \
    --encoder maskrcnn \
    --im_ratio original \
    --grid_size 8 --feat_dim 2048 \
    --distributed --multiGPU --mixed_precision \
    --update 1 \
    --clip_grad_norm 1.0 \

    # --vis_mask_predict \
    # --train mscoco_minival --valid mscoco_minival \
    # --train mscoco_train,mscoco_nominival,vgnococo --valid mscoco_minival \
    # --vis_mask_predict \
    # --loadLXMERT snap/pretrain/grid_cluster_lxrt_feat_Match_vocab10000/Epoch09
    # --visual_AR

    # --batchSize 100 \
    # --loadLXMERT snap/pretrain/grid_lxrt_feat_15/Epoch19
    # --batchSize 1000 \
     # --batchSize 560
    # --loadLXMERT snap/pretrain/vgbert/Epoch06
    # --fromScratch \
    # --taskMaskLM --taskObjPredict --taskMatched \
