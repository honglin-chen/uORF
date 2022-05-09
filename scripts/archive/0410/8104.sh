#!/bin/bash
DATAROOT=${1:-'room_chair_train'}
PORT=${2:-8104}
NSCENES=${3:-100}
python -m visdom.server -p $PORT &>/dev/null &
python train_without_gan.py --dataroot $DATAROOT --n_scenes $NSCENES --n_img_each_scene 4  \
    --checkpoints_dir 'checkpoints' --name '040722' \
    --display_port $PORT --display_ncols 4 --print_freq 200 --display_freq 200 --display_grad \
    --load_size 128 --n_samp 64 --input_size 64 --supervision_size 64 --coarse_epoch 600 --z_dim 64 --num_slots 2 \
    --model 'pixelnerf_nogan' \
    --gt_seg --pixel_nerf --pixel_decoder --pixel_encoder --no_use_background \
# done
echo "Done"
# error at epoch 600, after coarse epoch
# node 3 gpu 2