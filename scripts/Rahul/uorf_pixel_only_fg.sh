#!/bin/bash
DATAROOT=${1:-'tdw_multiview_texture'}
PORT=${2:-35205}
NSCENES=${3:-100}
python -m visdom.server -p $PORT &>/dev/null &
python train.py --dataroot $DATAROOT --n_scenes $NSCENES --n_img_each_scene 3  \
    --checkpoints_dir 'checkpoints' --name '35205' \
    --display_port $PORT --display_ncols 3 --print_freq 200 --display_freq 200 --display_grad \
    --load_size 128 --n_samp 128 --input_size 128 --mask_size 128 --supervision_size 64 \
    --niter 10000 --coarse_epoch 600 --z_dim 64 --num_slots 4 \
    --save_latest_freq 500 \
    --model 'uorf_train' \
    --focal_ratio 0.9605 0.9605 \
    --near_plane 1 --far_plane 15 \
    --unified_decoder \
    --no_locality_epoch 0 \
    --gt_seg \
    --pixel_encoder --mask_image \
    --use_ray_dir \
    --silhouette_loss --dice_loss \
    --bg_no_pixel \
    --continue_train --exp_id 'run-2022-05-05-01-51-59' \

#uORF + pixel features + silhoutte loss.

#    --frame5 \


#    --uorf \
#
#
#    --same_bg_fg_decoder \
#    --resnet_encoder \
#


# done
echo "Done"
# node 6 cuda 7
#    --silhouette_loss --dice_loss \
#    --mask_as_decoder_input --multiply_mask_pixelfeat \
#    --use_ray_dir --ray_after_density \