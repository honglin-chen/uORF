#!/bin/bash
DATAROOT=${1:-'tdw_multiview_texture'}
PORT=${2:-12011}
NSCENES=${3:-100}
python -m visdom.server -p $PORT &>/dev/null &
python train.py --dataroot $DATAROOT --n_scenes $NSCENES --n_img_each_scene 4  \
    --checkpoints_dir 'checkpoints' --name '12011' \
    --display_port $PORT --display_ncols 4 --print_freq 200 --display_freq 200 --display_grad \
    --load_size 128 --n_samp 64 --input_size 128 --supervision_size 64 --coarse_epoch 20 --z_dim 64 --num_slots 4 \
    --model 'uorf_train' \
    --focal_ratio 0.9605 0.9605 \
    --near_plane 1 --far_plane 30 \
    --no_locality_epoch 0 \
    --gt_seg --pixel_encoder --mask_image_feature --mask_image --use_ray_dir --silhouette_loss --weight_pixelfeat --small_latent --debug \
    --save_latest_freq 500 \
#    --continue_train --exp_id latest \

# done
echo "Done"
# bg no pixel is gone