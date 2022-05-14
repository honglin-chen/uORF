#!/bin/bash
DATAROOT=${1:-'tdw_30obj_multibg'}
PORT=${2:-19023}
NSCENES=${3:-3000}
python -m visdom.server -p $PORT &>/dev/null &
python train.py --dataroot $DATAROOT --n_scenes $NSCENES --n_img_each_scene 4  \
    --checkpoints_dir 'checkpoints' --name '19023' \
    --display_port $PORT --display_ncols 4 --print_freq 200 --display_freq 200 --display_grad \
    --load_size 128 --n_samp 128 --input_size 128 --mask_size 128 --supervision_size 64 \
    --niter 400 --coarse_epoch 200 --z_dim 64 --num_slots 4 \
    --save_latest_freq 500 \
    --model 'uorf_train' \
    --focal_ratio 0.9605 0.9605 \
    --near_plane 1 --far_plane 8 \
    --unified_decoder \
    --no_locality_epoch 0 \
    --gt_seg \
    --pixel_encoder --mask_image \
    --use_ray_dir \
    --restrict_world \
    --without_slot_feature \
    --pixel_nerf \
    --frame5 \
    --use_eisen_seg \
    --color_jitter \

echo "Done"
# node5 cuda 5