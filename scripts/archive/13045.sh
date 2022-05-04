#!/bin/bash
DATAROOT=${1:-'tdw_multiview_texture'}
PORT=${2:-13045}
NSCENES=${3:-100}
python -m visdom.server -p $PORT &>/dev/null &
python train.py --dataroot $DATAROOT --n_scenes $NSCENES --n_img_each_scene 4  \
    --checkpoints_dir 'checkpoints' --name '13045' \
    --display_port $PORT --display_ncols 4 --print_freq 200 --display_freq 200 --display_grad \
    --load_size 128 --n_samp 64 --input_size 128 --supervision_size 64 --niter 10000 --coarse_epoch 600 --z_dim 64 --num_slots 4 \
    --model 'uorf_train' \
    --focal_ratio 0.9605 0.9605 \
    --near_plane 1 --far_plane 15 \
    --no_locality_epoch 0 \
    --gt_seg --pixel_encoder --mask_image --use_ray_dir --silhouette_loss --bg_no_pixel \
    --save_latest_freq 500 \
    --mask_as_decoder_input --unified_decoder \
#    --continue_train --exp_id 'run-2022-04-24-16-41-21' \

# done
echo "Done"
# node 5 cuda 3

# this is for testing the following:
# (1) remove --mask_image_feature /
# (2) put density as an input of color decoder instead of masks
# (3) reduce color decoder for memory issue
# (4) put mask as an input of both color decoder and density decoder
# density is different from masks
# we might also need masks, because it is not provided from image_feature in this case.
# or, density might be more reasonable input data. mask might again has too strong opinion
# the point of letting cnn receptive field do this job is making it learnable (which does not necessarily mean that it will actually learn)

# if transmittance and ray_dir is not important, it is possible to merge the color decoder and density decoder.
# but be aware that if we reduce the color decoder well, this is not necessary.
# in principle, distinguish color decoder and density decoder, even if the difference is only ray_dir, is great thing to do.
# because density should not be affected by ray_dir, but color should be affected by it.

# for performance, we might need to multiply mask and pixel feat, but let's not do that right now for memory issue

# (2) is not well-thought. we need to think more
#