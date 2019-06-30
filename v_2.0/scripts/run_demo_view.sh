# http://10.10.10.100
# 31100

# http://10.19.125.13
# 10087
#    --data_dir  /public/liuwen/p300/human_pose/processed \
#    --place_dir  /public/liuwen/p300/places365_standard \
#    --checkpoints_dir  /public/liuwen/p300/models \


# 6 epoch
#python demo_view.py --gpu_ids 7 \
#    --model viewer \
#    --gen_name impersonator \
#    --dataset_mode  mi_v2 \
#    --name impersonator_02 \
#    --output_dir /p300/poseGANs \
#    --image_size 256  \
#    --map_name uv_seg   \
#    --view_params R=0,180,0/t=0,0,0 \
#    --src_path /home/piaozx/liuwen/p300/results/impersonator_02_14/009_5_1/000.jpg    \
#    --tgt_path /home/piaozx/liuwen/p300/results/impersonator_02_14/023_3_1/000.jpg  --visual \
#    --ip http://10.19.125.13 --port 10087   \
#    --data_dir  /public/liuwen/p300/human_pose/processed \
#    --checkpoints_dir  /public/liuwen/p300/models

python demo_view.py --gpu_ids 7 \
    --model viewer \
    --gen_name impersonator \
    --dataset_mode  mi_v2 \
    --name impersonator_02 \
    --output_dir /p300/poseGANs \
    --image_size 256  \
    --map_name uv_seg   \
    --view_params R=0,180,0/t=0,0,0 \
    --src_path /home/piaozx/liuwen/p300/results/impersonator_02_14/024_7_2/0000.jpg    \
    --tgt_path /home/piaozx/liuwen/p300/results/impersonator_02_14/023_3_1/000.jpg  --visual \
    --ip http://10.19.125.13 --port 10087   \
    --data_dir  /public/liuwen/p300/human_pose/processed \
    --checkpoints_dir  /public/liuwen/p300/models

