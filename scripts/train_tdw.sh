#!/bin/bash
DATAROOT=${1:-'playroom_v0_train'}
PORT=${2:-8013}
python -m visdom.server -p $PORT &>/dev/null &
python train_without_gan.py --dataroot $DATAROOT --n_scenes 40000 --n_img_each_scene 4  \
    --checkpoints_dir 'checkpoints' --name 'room_chair' \
    --display_port $PORT --display_ncols 4 --print_freq 100 --display_freq 200 --display_grad \
    --load_size 128 --n_samp 64 --input_size 64 --supervision_size 64 --coarse_epoch 600 --z_dim 64 --num_slots 4 --batch_size 8 \
    --model 'uorf_nogan' --gt_seg --frame5 \
# done
echo "Done"
