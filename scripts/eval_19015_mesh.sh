#!/bin/bash
DATAROOT=${1:-'tdw_30obj_multibg'}
CHECKPOINT=${2:-'/data2/honglinc/uORF/checkpoints/'}
PORT=8078
python -m visdom.server -p $PORT &>/dev/null &
python test.py --dataroot $DATAROOT --n_scenes 600 --n_img_each_scene 4 \
    --checkpoints_dir $CHECKPOINT --name '19015' --exp_id 'latest' \
    --results_dir 'results' \
    --display_port $PORT --display_ncols 4 \
    --load_size 256 --input_size 128 --mask_size 256 --render_size 32 --frustum_size 256 \
    --n_samp 256 --z_dim 64 --num_slots 4 \
    --model 'uorf_eval' \
    --skip 3049 \
    --focal_ratio 0.9605 0.9605 \
    --near_plane 1 --far_plane 8 \
    --unified_decoder \
    --no_locality_epoch 0 \
    --gt_seg \
    --pixel_encoder --mask_image \
    --use_ray_dir \
    --restrict_world \
    --frame5 \
    --use_eisen_seg --extract_mesh \

echo "Done"

