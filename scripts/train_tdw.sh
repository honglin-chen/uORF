#!/bin/bash
DATAROOT=${1:-'tdw_uorf'}
PORT=${2:-8011}
python -m visdom.server -p $PORT &>/dev/null &
python train_without_gan.py --dataroot $DATAROOT --n_scenes 1 --n_img_each_scene 4  \
    --checkpoints_dir 'checkpoints' --name 'tdw_uorf' \
    --display_port $PORT --display_ncols 4 --print_freq 200 --display_freq 200 --display_grad \
    --load_size 128 --n_samp 64 --input_size 64 --supervision_size 64 --z_dim 64 --num_slots 5 \
    --nss_scale 10 --obj_scale 8 --fixed_locality --niter 6000 --coarse_epoch 3000 \
    --model 'uorf_nogan' \
# done
echo "Done"
