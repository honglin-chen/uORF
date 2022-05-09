#!/bin/bash
DATAROOT=${1:-'tdw_zoo_10obj_2000'}
CHECKPOINT=${2:-'/data2/honglinc/uORF/checkpoints/'}
PORT=8077
python -m visdom.server -p $PORT &>/dev/null &
python test.py --dataroot $DATAROOT --n_scenes 10 --n_img_each_scene 4 \
    --checkpoints_dir $CHECKPOINT --name '14240' --exp_id 'run-2022-05-01-21-48-38' \
    --results_dir 'results' \
    --display_port $PORT --display_ncols 4 \
    --load_size 256 --input_size 256 --render_size 8 --frustum_size 128 \
    --n_samp 128 --z_dim 64 --num_slots 4 \
    --model 'uorf_eval' \
    --skip 100 \
    --focal_ratio 0.9605 0.9605 \
    --near_plane 1 --far_plane 8 \
    --gt_seg --pixel_encoder --mask_image \
    --restrict_world \
    --unified_decoder \
    --use_ray_dir \
    --extract_mesh
#    --debug2 \
#    --pixel_after_density \
#    --without_slot_feature \

# same bg fg decoder did not work before
echo "Done"