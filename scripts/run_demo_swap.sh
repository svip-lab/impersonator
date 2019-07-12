
# 6 epoch
python demo_swap.py --gpu_ids 2 \
    --model swapper \
    --gen_name impersonator \
    --dataset_mode  mi_v2 \
    --name impersonator_02 \
    --output_dir /p300/poseGANs \
    --image_size 256  \
    --map_name uv_seg   \
    --swap_part body \
    --src_path /p300/poseGANs/impersonator_02_14/024_7_2/0000.jpg    \
    --tgt_path /p300/poseGANs/impersonator_02_14/023_3_1/000.jpg  --visual



