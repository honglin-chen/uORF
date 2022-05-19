#!/bin/bash
DATAROOT=${1:-'tdw_zoo_10obj_2000'}
PORT=${2:-17005}
NSCENES=${3:-1000}
python -m visdom.server -p $PORT &>/dev/null &
python train.py --dataroot $DATAROOT --n_scenes $NSCENES --n_img_each_scene 4  \
    --checkpoints_dir 'checkpoints' --name '17005' \
    --display_port $PORT --display_ncols 4 --print_freq 200 --display_freq 200 --display_grad \
    --load_size 128 --n_samp 128 --input_size 128 --mask_size 128 --supervision_size 64 \
    --niter 1200 --coarse_epoch 600 --z_dim 64 --num_slots 4 \
    --save_latest_freq 500 \
    --model 'uorf_train' \
    --focal_ratio 0.9605 0.9605 \
    --near_plane 1 --far_plane 8 \
    --unified_decoder \
    --no_locality_epoch 0 \
    --gt_seg \
    --restrict_world \
    --frame5 \
    --resnet_encoder \
    --use_ray_dir --ray_after_density \
    --pixel_encoder --mask_image --color_after_density \
    --silhouette_loss --silhouette_l2_loss \

echo "Done"
# node5 cuda 3