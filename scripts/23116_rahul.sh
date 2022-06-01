#!/bin/bash
DATAROOT=${1:-'tdw_physion_two_domino_elev_fixed'}
PORT=${2:-23116}
NSCENES=${3:-200}
python -m visdom.server -p $PORT &>/dev/null &
python train.py --dataroot $DATAROOT --n_scenes $NSCENES --n_img_each_scene 4  \
    --checkpoints_dir 'checkpoints' --name '23116_rahul' \
    --display_port $PORT --display_ncols 4 --print_freq 200 --display_freq 200 --display_grad \
    --load_size 128 --n_samp 128 --input_size 128 --mask_size 128 --supervision_size 64 \
    --niter 1200 --coarse_epoch 600 --z_dim 64 --num_slots 3 \
    --save_latest_freq 5000 \
    --model 'uorf_train' \
    --focal_ratio 0.9605 0.9605 \
    --near_plane 1 --far_plane 15 \
    --unified_decoder \
    --no_locality_epoch 0 \
    --gt_seg \
    --pixel_encoder --mask_image \
    --bg_no_pixel \
    --use_ray_dir --debug_no_ray_dir \
    --percept_in 0 \
    --continue_train --exp_id 'latest' \
    --silhouette_loss --silhouette_l2_loss --bg_no_silhouette_loss \

#    --silhouette_loss --silhouette_l2_loss --bg_no_silhouette_loss --silhouette_expand --fg_only_delete_bg \
#    --continue_train --exp_id latest \


#   --use_ray_dir \
#    --continue_train --exp_id 'latest' \

echo "Done"
# 25
# node 6 gpu 6
# node 6 gpu 3 --> 19916 same thing
