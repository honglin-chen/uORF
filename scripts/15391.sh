#!/bin/bash
DATAROOT=${1:-'tdw_zoo_10obj_2000'}
PORT=${2:-15391}
NSCENES=${3:-10}
python -m visdom.server -p $PORT &>/dev/null &
python train.py --dataroot $DATAROOT --n_scenes $NSCENES --n_img_each_scene 4  \
    --checkpoints_dir 'checkpoints' --name '15391' \
    --display_port $PORT --display_ncols 4 --print_freq 20 --display_freq 2 --display_grad \
    --load_size 128 --n_samp 48 --input_size 128 --mask_size 128 --supervision_size 64 \
    --niter 120 --coarse_epoch 120 --z_dim 64 --num_slots 4 \
    --save_latest_freq 500 \
    --model 'uorf_train' \
    --focal_ratio 0.9605 0.9605 \
    --near_plane 1 --far_plane 8 \
    --unified_decoder \
    --no_locality_epoch 0 \
    --gt_seg \
    --restrict_world \
    --frame5 \
    --resnet_encoder --predict_centroid --slot_positional_encoding \
    --use_ray_dir --ray_after_density \
    --pixel_encoder --mask_image --color_after_density \
#    --learn_fg_first --learn_bg_next \
#    --silhouette_loss --progressive_silhouette --dice_loss \
#    --learn_fg_first \
#    --silhouette_loss --progressive_silhouette --dice_loss \
#    --continue_train --exp_id 'latest'

#        --silhouette_loss --silhouette_l2_loss --silhouette_epoch 0 \

#    --pixel_encoder --mask_image \
#    --use_ray_dir \
#    --same_bg_fg_decoder \
#
#    --restrict_world \


#    --continue_train --exp_id 'run-2022-04-24-16-41-21' \

# done
echo "Done"
# node 3 cuda 7
#    --silhouette_loss --dice_loss \
#    --mask_as_decoder_input --multiply_mask_pixelfeat \
#    --use_ray_dir --ray_after_density \