#--src_path /home/piaozx/liuwen/p300/imgs/3.jpg    \
#--tgt_path /home/piaozx/liuwen/p300/human_pose/processed/motion_transfer_HD/001/1/2/0640.jpg  --visual \

python demo_inpaintor.py --gpu_ids 8 \
    --model background_inpaintor \
    --name background_inpaintor \
    --image_size 256  \
    --map_name uv_seg   \
    --src_path /home/piaozx/liuwen/p300/imgs/3.jpg \
    --ip http://10.19.126.34 \
    --port 10087 \
    --checkpoints_dir /home/piaozx/liuwen/p300/models   \
    --visual

# /home/piaozx/liuwen/p300/imgs/3.jpg

