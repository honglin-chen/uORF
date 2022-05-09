#!/bin/bash
DATAROOT=${1:-'tdw_zoo_10obj'}
PORT=${2:-14241}
NSCENES=${3:-100}
python -m visdom.server -p $PORT &>/dev/null &
python train.py --dataroot $DATAROOT --n_scenes $NSCENES --n_img_each_scene 4  \
    --checkpoints_dir 'checkpoints' --name '14241' \
    --display_port $PORT --display_ncols 4 --print_freq 200 --display_freq 200 --display_grad \
    --load_size 256 --n_samp 128 --input_size 256 --mask_size 256 --supervision_size 64 \
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
    --silhouette_loss --dice_loss \
    --continue_train --exp_id 'run-2022-05-01-21-48-38' \
    --frame5 \

#    --same_bg_fg_decoder \

# done
echo "Done"
# node 5 cuda 5
# node 6 cuda 3
#    --silhouette_loss --dice_loss \
#    --mask_as_decoder_input --multiply_mask_pixelfeat \
#    --use_ray_dir --ray_after_density \