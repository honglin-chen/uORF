#!/bin/bash
DATAROOT=${1:-'tdw_zoo_10obj_2000'}
PORT=${2:-15374}
NSCENES=${3:-1}
python -m visdom.server -p $PORT &>/dev/null &
python train.py --dataroot $DATAROOT --n_scenes $NSCENES --n_img_each_scene 4  \
    --checkpoints_dir 'checkpoints' --name '15374' \
    --display_port $PORT --display_ncols 4 --print_freq 20 --display_freq 20 --display_grad \
    --load_size 128 --n_samp 128 --input_size 128 --mask_size 128 --supervision_size 64 \
    --niter 1200 --coarse_epoch 600 --z_dim 64 --num_slots 4 \
    --save_latest_freq 50 \
    --model 'uorf_train' \
    --focal_ratio 0.9605 0.9605 \
    --near_plane 1 --far_plane 8 \
    --unified_decoder \
    --no_locality_epoch 0 \
    --gt_seg \
    --uorf \
    --restrict_world \
    --frame5 \
    --silhouette_loss --dice_loss --silhouette_epoch 0 --learn_only_silhouette --bg_no_silhouette_loss \
    --resnet_encoder \
    --continue_train --exp_id 'latest' \

# copy of 51
# node3 gpu2
#    --pixel_encoder --mask_image \
#    --use_ray_dir \
#    --same_bg_fg_decoder \
#
#    --restrict_world \


#    --continue_train --exp_id 'run-2022-04-24-16-41-21' \

# done
echo "Done"
# node 6 cuda 1
#    --silhouette_loss --dice_loss \
#    --mask_as_decoder_input --multiply_mask_pixelfeat \
#    --use_ray_dir --ray_after_density \