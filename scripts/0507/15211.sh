#!/bin/bash
DATAROOT=${1:-'tdw_zoo_10obj'}
PORT=${2:-15211}
NSCENES=${3:-100}
python -m visdom.server -p $PORT &>/dev/null &
python train.py --dataroot $DATAROOT --n_scenes $NSCENES --n_img_each_scene 4  \
    --checkpoints_dir 'checkpoints' --name '15211' \
    --display_port $PORT --display_ncols 4 --print_freq 200 --display_freq 200 --display_grad \
    --load_size 128 --n_samp 128 --input_size 128 --mask_size 128 --supervision_size 64 \
    --niter 10000 --coarse_epoch 600 --z_dim 64 --num_slots 4 \
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
    --frame5 \
    --color_after_density \

# if not same fg bg decoder, color after density is also means bg_no_pixel
#    --uorf \
#
#
#    --same_bg_fg_decoder \
#    --resnet_encoder \
#


#    --continue_train --exp_id 'run-2022-04-24-16-41-21' \

# done
echo "Done"
# node 6 cuda 4
#    --silhouette_loss --dice_loss \
#    --mask_as_decoder_input --multiply_mask_pixelfeat \
#    --use_ray_dir --ray_after_density \