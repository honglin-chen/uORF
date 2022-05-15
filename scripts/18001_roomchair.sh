#!/bin/bash
DATAROOT=${1:-'room_chair_train'}
PORT=${2:-18391}
NSCENES=${3:-100}
python -m visdom.server -p $PORT &>/dev/null &
python train_without_gan.py --dataroot $DATAROOT --n_scenes $NSCENES --n_img_each_scene 4  \
    --checkpoints_dir 'checkpoints' --name '18391' \
    --display_port $PORT --display_ncols 4 --print_freq 200 --display_freq 20 --display_grad \
    --load_size 128 --n_samp 128 --input_size 128 --mask_size 128 --supervision_size 64 \
    --niter 20000 --coarse_epoch 10000 --z_dim 64 --num_slots 5 \
    --save_latest_freq 500 \
    --model 'uorf_train' \
    --near_plane 6 --far_plane 20 \
    --unified_decoder \
    --no_locality_epoch 300 \
    --uorf \

echo "Done"
# 34
# node 6 gpu 1