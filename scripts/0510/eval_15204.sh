#!/bin/bash
DATAROOT=${1:-'tdw_zoo_10obj'}
CHECKPOINT=${2:-'/data2/wanhee/uORF/checkpoints/'}
PORT=8077
python -m visdom.server -p $PORT &>/dev/null &
python test.py --dataroot $DATAROOT --n_scenes 10 --n_img_each_scene 4 \
    --checkpoints_dir $CHECKPOINT --name '15204' --exp_id 'run-2022-05-03-01-18-38' \
    --results_dir 'results' \
    --display_port $PORT --display_ncols 4 \
    --load_size 256 --input_size 256 --render_size 8 --frustum_size 128 \
    --n_samp 128 --z_dim 64 --num_slots 2 \
    --model 'uorf_eval' \
    --skip 100 \
    --focal_ratio 0.9605 0.9605 \
    --near_plane 1 --far_plane 8 \
    --unified_decoder \
    --gt_seg \
    --pixel_encoder --mask_image \
    --use_ray_dir \
    --restrict_world \
    --combine_masks \
    --without_slot_feature \
    --pixel_nerf \
    --frame5 \

#    --debug2 \
#    --pixel_after_density \
#    --without_slot_feature \
# /data2/wanhee/uORF/checkpoints/15201/run-2022-05-03-01-02-10 node6
#/data2/wanhee/uORF/checkpoints/15202/run-2022-05-03-00-28-03 node6
#/data2/wanhee/uORF/checkpoints/15202/run-2022-05-03-01-05-51 node6
#/data2/wanhee/uORF/checkpoints/15203/run-2022-05-03-01-14-20 node6
#/data2/wanhee/uORF/checkpoints/15204/run-2022-05-03-01-18-38 node6
# same bg fg decoder did not work before
echo "Done"
#render size was 8
#n_samp was 256
# inputsize was 128
# frustum size was 128
#    --checkpoints_dir $CHECKPOINT --name '13000_backup' --exp_id 'run-2022-04-24-16-41-21' --results_dir 'results' \
# --name '14081' --exp_id 'run-2022-05-01-21-44-20'
# --name '14181' --exp_id 'run-2022-05-02-00-00-26' tdw_multiview_10obj
# --name '14140' --exp_id 'run-2022-05-01-00-45-32'
# --name '14040' --exp_id 'run-2022-05-01-00-36-30' tdw_multiview_texture