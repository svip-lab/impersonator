python demo_animate.py --gpu_ids 2 \
    --model animator \
    --gen_name impersonator \
    --dataset_mode  mi_v2 \
    --name impersonator_02 \
    --output_dir '' \
    --checkpoints_dir /home/piaozx/liuwen/p300/models \
    --image_size 256  \
    --map_name uv_seg   \
    --swap_part body \
    --ip http://10.19.125.13 --port 10087   \
    --src_path /home/piaozx/liuwen/p300/results/impersonator_02_29/024_7_2/0000.jpg    \
    --ref_path /home/piaozx/liuwen/p300/results/impersonator_02_29/023_3_1/000.jpg  \
    --tgt_path /home/piaozx/liuwen/p300/human_pose/processed/motion_transfer_HD/001/1/2/0640.jpg  \
    --visual



