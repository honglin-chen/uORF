#!/bin/bash
DATAROOT=${1:-'tdw_20obj_2000'}
PORT=${2:-8011}
python -m visdom.server -p $PORT &>/dev/null &
python train.py --dataroot $DATAROOT --n_scenes 10 --n_img_each_scene 4  \
    --checkpoints_dir 'checkpoints' --name 'debug' \
    --display_port $PORT --display_ncols 4 --print_freq 2 --display_freq 2 --display_grad \
    --load_size 128 --n_samp 64 --input_size 128 --supervision_size 64 --batch_size 2 \
    --niter 1200 --coarse_epoch 600 --z_dim 64 --num_slots 4 \
    --save_latest_freq 5000 \
    --model 'uorf_train' \
    --no_locality_epoch 0 \
    --percept_in 100 \
    --focal_ratio 0.9605 0.9605 \
    --near_plane 1 --far_plane 15 \
    --use_slot_feat --use_pixel_feat --gt_seg --frame5 --mask_image \
    --visualize_unoccl_silhouette --visualize_occl_silhouette --visualize_mask \
    --use_occl_silhouette_loss \
    --unisurf_render_eq \
    --continue_train --exp_id 'latest' \

# done
echo "Done"
