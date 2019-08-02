python evaluate_apt.py --gpu_ids 0 \
    --model swapper \
    --gen_name impersonator \
    --dataset_mode  mi_v2 \
    --name impersonator_02 \
    --output_dir /p300/poseGANs/appearance \
    --image_size 256  \
    --swap_part body \
    --map_name uv_seg --cam_strategy source \
    --visual
