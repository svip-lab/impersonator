python test.py --gpu_ids 2 \
    --model impersonator_01 \
    --dataset_mode  fast_mi \
    --name impersonator_01 \
    --image_size 256  \
    --map_name uv_seg \
    --visual


# 6 epoch
python test.py --gpu_ids 8 \
    --data_dir  /public/liuwen/p300/human_pose/processed \
    --place_dir  /public/liuwen/p300/places365_standard \
    --checkpoints_dir  /public/liuwen/p300/models \
    --model impersonator \
    --name impersonator \
    --dataset_mode  mi_v2 \
    --image_size 256  \
    --map_name uv_seg \
    --train_ids_file  MI_val_debug.txt  \
    --test_ids_file   MI_val_debug.txt  \
    --visual


