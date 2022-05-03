#!/bin/bash
DATAROOT=${1:-'tdw_multiview_10obj'}
PORT=${2:-14181}
NSCENES=${3:-100}
python -m visdom.server -p $PORT &>/dev/null &
python train.py --dataroot $DATAROOT --n_scenes $NSCENES --n_img_each_scene 4  \
    --checkpoints_dir 'checkpoints' --name '14181' \
    --display_port $PORT --display_ncols 4 --print_freq 200 --display_freq 200 --display_grad \
    --load_size 256 --n_samp 96 --input_size 256 --mask_size 256 --supervision_size 64 \
    --niter 10000 --coarse_epoch 600 --z_dim 64 --num_slots 4 \
    --model 'uorf_train' \
    --focal_ratio 0.9605 0.9605 \
    --near_plane 1 --far_plane 8 \
    --no_locality_epoch 0 \
    --gt_seg --pixel_encoder --mask_image \
    --restrict_world \
    --save_latest_freq 500 \
    --unified_decoder \
    --use_ray_dir \
    --same_bg_fg_decoder \
    --without_slot_feature \


#    --continue_train --exp_id 'run-2022-04-24-16-41-21' \

# done
echo "Done"
# node 5 cuda 3
#    --silhouette_loss --dice_loss \
#    --mask_as_decoder_input --multiply_mask_pixelfeat \
#    --use_ray_dir --ray_after_density \