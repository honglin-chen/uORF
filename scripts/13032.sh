#!/bin/bash
DATAROOT=${1:-'tdw_multiview_texture'}
PORT=${2:-13032}
NSCENES=${3:-100}
python -m visdom.server -p $PORT &>/dev/null &
python train.py --dataroot $DATAROOT --n_scenes $NSCENES --n_img_each_scene 4  \
    --checkpoints_dir 'checkpoints' --name '13032' \
    --display_port $PORT --display_ncols 4 --print_freq 200 --display_freq 200 --display_grad \
    --load_size 128 --n_samp 64 --input_size 128 --supervision_size 64 --niter 10000 --coarse_epoch 600 --z_dim 64 --num_slots 4 \
    --model 'uorf_train' \
    --focal_ratio 0.9605 0.9605 \
    --near_plane 1 --far_plane 15 \
    --no_locality_epoch 0 \
    --gt_seg --pixel_encoder --mask_image_feature --mask_image --use_ray_dir --silhouette_loss --weight_pixelfeat --density_no_pixel \
    --save_latest_freq 500 \
    --restrict_world \
#    --continue_train --exp_id 'run-2022-04-24-16-41-21' \

# done
echo "Done"
# node 5 cuda 2
# 1304x series are for learning density from only slot features, excluding pixel features.
# 13030.sh, which naively removes pixel feature from did not work well. (did not learn at all.)
# There are a few ways to try this.
# 13031.sh: previously we observed that this should work. Therefore, we need to reproduce this by having rbg loss from density decoder
# 13032.sh: Another problem we observed from extracted mesh is that the world and the objects are very small compared to our sampling size. this might cause problems on background, etc. Therefore, restricting world might help
# 13033.sh: similar to 13041.sh. restricting z_near and z_far would be another way to achieve the purpose of 13041.sh.
# 13034.sh: training only silhouette loss first can make us understand this problem better. (is rgb important for learning geometry? or silhouette loss is enough?)