#!/bin/bash
DATAROOT=${1:-'tdw_multiview_texture'}
PORT=${2:-10066}
NSCENES=${3:-100}
python -m visdom.server -p $PORT &>/dev/null &
python train_without_gan.py --dataroot $DATAROOT --n_scenes $NSCENES --n_img_each_scene 4  \
    --checkpoints_dir 'checkpoints' --name '10067' \
    --display_port $PORT --display_ncols 4 --print_freq 200 --display_freq 200 --display_grad \
    --load_size 128 --n_samp 64 --input_size 64 --supervision_size 64 --coarse_epoch 600 --z_dim 64 --num_slots 5 \
    --model 'uorf_nogan' \
    --focal_ratio 0.9605 0.9605 \
    --near_plane 1 --far_plane 15 \
    --gt_seg --pixel_encoder --mask_image_feature --mask_image --bg_no_pixel --use_ray_dir --weigh_pixelfeat --div_by_max --silhouette_loss \
# done
echo "Done"
# node 5 cuda 5
# silhouette till 300
# changed to ... (l2->l1, [...]->[1:, ...]) : I think this is not good
# self.loss_silhouette = self.L1_loss(silhouette[1:, ...], silhouette_masks[1:, ...])