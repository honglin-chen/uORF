#!/bin/bash
DATAROOT=${1:-'tdw_multiview_texture'}
CHECKPOINT=${2:-'./checkpoints'}
PORT=${3:-35216}
NSCENES=${4:-1}
python -m visdom.server -p $PORT &>/dev/null &
python train.py --dataroot $DATAROOT --n_scenes $NSCENES --n_img_each_scene 4  \
    --checkpoints_dir $CHECKPOINT --name '35214_feat_full_data_no_ray_dir' \
    --display_port $PORT --display_ncols 4 --print_freq 200 --display_freq 40 --display_grad \
    --load_size 128 --n_samp 128 --input_size 128 --mask_size 128 --supervision_size 64 \
    --nss_scale 1 \
    --niter 10000 --coarse_epoch 30000 --z_dim 64 --num_slots 3 \
    --save_latest_freq 500 \
    --model 'uorf_train' \
    --focal_ratio 0.9605 0.9605 \
    --near_plane 2 --far_plane 7 \
    --unified_decoder \
    --no_locality_epoch 0 \
    --gt_seg \
    --pixel_encoder --mask_image \
    --silhouette_loss --silhouette_l2_loss_masked \
    --bg_no_pixel \
    --continue_train --exp_id 'run-2022-05-20-01-32-00' \
#    --use_ray_dir \
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