#!/bin/bash
DATAROOT=${1:-'tdw_multiview_texture'}
CHECKPOINT=${2:-'/data2/wanhee/uORF/checkpoints/'}
PORT=8077
python -m visdom.server -p $PORT &>/dev/null &
python test.py --dataroot $DATAROOT --n_scenes 1 --n_img_each_scene 4 \
    --checkpoints_dir $CHECKPOINT --name 10031 --exp_id latest --results_dir 'results' \
    --display_port $PORT --display_ncols 4 \
    --load_size 128 --input_size 64 --render_size 8 --frustum_size 128 \
    --n_samp 256 --z_dim 64 --num_slots 5 \
    --model 'uorf_eval' \
    --focal_ratio 0.9605 0.9605 \
    --near_plane 1 --far_plane 20 \
    --gt_seg --pixel_encoder --mask_image_feature --mask_image --bg_no_pixel --use_ray_dir \

echo "Done"
