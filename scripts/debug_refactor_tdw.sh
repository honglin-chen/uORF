#!/bin/bash
DATAROOT=${1:-'tdw_20obj_2000'}
PORT=${2:-8011}
python -m visdom.server -p $PORT &>/dev/null &
python train.py --dataroot $DATAROOT --n_scenes 10 --n_img_each_scene 4  \
    --checkpoints_dir 'checkpoints' --name 'debug' \
    --display_port $PORT --display_ncols 4 --print_freq 2 --display_freq 2 --display_grad \
    --load_size 128 --n_samp 64 --input_size 128 --supervision_size 64 --coarse_epoch 600 --z_dim 64 --num_slots 4 --batch_size 2 \
    --model 'uorf_train' \
    --use_pixel_feat --use_slot_feat --gt_seg --frame5 \
# done
echo "Done"
