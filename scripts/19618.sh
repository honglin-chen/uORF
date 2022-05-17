#!/bin/bash
DATAROOT=${1:-'tdw_30obj_multibg'}
PORT=${2:-19618}
NSCENES=${3:-3000}
python -m visdom.server -p $PORT &>/dev/null &
python train.py --dataroot $DATAROOT --n_scenes $NSCENES --n_img_each_scene 4  \
    --checkpoints_dir 'checkpoints' --name '19618_2' \
    --display_port $PORT --display_ncols 4 --print_freq 200 --display_freq 200 --display_grad \
    --load_size 128 --n_samp 128 --input_size 128 --mask_size 128 --supervision_size 64 \
    --niter 16 --coarse_epoch 20 --z_dim 64 --num_slots 2 \
    --save_latest_freq 500 \
    --model 'uorf_train' \
    --focal_ratio 0.9605 0.9605 \
    --near_plane 1 --far_plane 8 \
    --unified_decoder \
    --no_locality_epoch 0 \
    --gt_seg \
    --pixel_encoder --mask_image \
    --combine_masks --dataset_combine_masks \
    --use_ray_dir \
    --restrict_world \
    --frame5 \
    --continue_train --exp_id 'latest' \
    --percept_in 0 \
#    --border_zero \
#    --silhouette_loss --silhouette_l2_loss --bg_no_silhouette_loss --fg_only_delete_bg --stop_hungarian \


echo "Done"
# node5 cuda 5