#!/bin/bash
DATAROOT=${1:-'/data2/wanhee/uORF/tdw_physion_two_domino_elev_fixed/'}
CHECKPOINT=${2:-'/data2/wanhee/uORF/checkpoints/'}
PORT=8078
python -m visdom.server -p $PORT &>/dev/null &
python test.py --dataroot $DATAROOT --n_scenes 10 --n_img_each_scene 4 \
    --checkpoints_dir $CHECKPOINT --name '23116_rahul' --exp_id 'latest' \
    --results_dir 'results' \
    --display_port $PORT --display_ncols 4 \
    --load_size 128 --input_size 128 --mask_size 128 --render_size 32 --frustum_size 128 \
    --n_samp 128 --z_dim 64 --num_slots 3 \
    --model 'uorf_eval' \
    --skip 0 \
    --focal_ratio 0.9605 0.9605 \
    --near_plane 1 --far_plane 15 \
    --unified_decoder \
    --no_locality_epoch 0 \
    --gt_seg \
    --pixel_encoder --mask_image \
    --use_ray_dir --debug_no_ray_dir \
    --bg_no_pixel \
    --extract_mesh \
    --save_mesh_xyz_density \
#    --extract_mesh_camproj \
#


echo "Done"

