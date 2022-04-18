#!/bin/bash
DATAROOT=${1:-'tdw_multiview_chair'}
PORT=${2:-9100}
NSCENES=${3:-100}
python -m visdom.server -p $PORT &>/dev/null &
python train_without_gan.py --dataroot $DATAROOT --n_scenes $NSCENES --n_img_each_scene 4  \
    --checkpoints_dir 'checkpoints' --name '041222' \
    --display_port $PORT --display_ncols 4 --print_freq 200 --display_freq 200 --display_grad \
    --load_size 128 --n_samp 64 --input_size 64 --supervision_size 64 --coarse_epoch 600 --z_dim 64 --num_slots 5 \
    --model 'uorf_nogan' \
    --focal_ratio 0.9605 0.9605 \
    --near_plane 2 --far_plane 8 \
    --gt_seg --pixel_encoder --mask_image_feature \
# done
echo "Done"
# node 3 cuda 4