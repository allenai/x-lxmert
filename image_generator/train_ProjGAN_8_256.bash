python main.py \
    --clustering \
    --im_ratio='original' \
    --resize_input_size=512 \
    --resize_target_size=256 \
    --n_grid=8 --emb_dim=2048 \
    --g_base_dim=32 \
    --d_base_dim=64 \
    --classifier='resnet50' \
    --encoder='maskrcnn' \
    --gan --gan_loss_lambda=1 --gan_feat_match_lambda=10 \
    --recon_loss_lambda=0 --feat_loss_lambda=10 \
    --hinge \
    --g_norm_type='spade_in' \
    --multiGPU --distributed \
    --batch_size=16 \
    --epochs=51 \
    --g_adam_beta1 0.0 \
    --d_adam_beta1 0.0 \
    --all_layers \
    --SN \
    --gan_loss_cluster_lambda=1 \
    --workers 8
    # --ACGAN \
