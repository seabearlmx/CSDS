#!/usr/bin/env bash
    # Example on Cityscapes
    # --val_dataset gtav bdd100k cityscapes synthia mapillary \
     python -m torch.distributed.launch --nproc_per_node=1 get_class_patches.py \
        --dataset cityscapes \
        --covstat_val_dataset cityscapes \
        --val_dataset cityscapes \
        --arch network.deepv3.DeepR50V3PlusD \
        --city_mode 'train' \
        --lr_schedule poly \
        --lr 0.01 \
        --poly_exp 0.9 \
        --max_cu_epoch 10000 \
        --class_uniform_pct 0.5 \
        --class_uniform_tile 1024 \
        --crop_size 768 \
        --scale_min 0.5 \
        --scale_max 2.0 \
        --rrotate 0 \
        --max_iter 60000 \
        --bs_mult 1 \
        --gblur \
        --color_aug 0.5 \
        --wt_reg_weight 0.0 \
        --relax_denom 0.0 \
        --cov_stat_epoch 0 \
        --wt_layer 0 0 4 4 4 0 0 \
        --date 0101 \
        --exp r50os16_cityscapes_ibn \
        --ckpt ./logs/ \
        --tb_path ./logs/
