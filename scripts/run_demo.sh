python demo.py --gpu_ids 2 \
    --model impersonator_01 \
    --dataset_mode  fast_mi \
    --name impersonator_01 \
    --image_size 256  \
    --map_name uv_seg


# 6 epoch
python demo.py --gpu_ids 9 \
    --model imitator \
    --gen_name impersonator \
    --dataset_mode  mi_v2 \
    --name impersonator_02 \
    --output_dir  '' \
    --image_size 256  \
    --map_name uv_seg   \
    --src_path /public/liuwen/p300/imgs/3.jpg    \
    --tgt_path /public/liuwen/p300/ImPer/motion_transfer_HD/024/8/2/0695.jpg  --visual \
    --ip http://10.19.129.76 \
    --port 10086 \
    --checkpoints_dir /public/liuwen/p300/models --do_saturate_mask  \
    --bg_replace


--src_path /public/liuwen/p300/imgs/3.jpg    \
--src_path test_images/men_1_input.png    \


python demo.py --gpu_ids 9 \
    --model imitator_v2 \
    --gen_name impersonator \
    --dataset_mode  mi_v2 \
    --name impersonator_02 \
    --output_dir  '' \
    --image_size 256  \
    --map_name uv_seg   \
    --src_path /public/liuwen/p300/imgs/8.jpg    \
    --tgt_path /public/liuwen/p300/ImPer/motion_transfer_HD/024/8/2/0695.jpg  --visual \
    --ip http://10.19.129.76 \
    --port 10086 \
    --checkpoints_dir /public/liuwen/p300/models  --has_detector  --bg_ks 7

# /home/piaozx/liuwen/p300/imgs/3.jpg

