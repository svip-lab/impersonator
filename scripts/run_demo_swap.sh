
# AI DOCKER, impersonator in iccv
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


# RTX 01, impersonator v3
python demo_swap.py --gpu_ids 9 \
    --model swapper_v2 \
    --gen_name impersonator \
    --name impersonator_mi \
    --checkpoints_dir /public/liuwen/p300/models \
    --output_dir '' \
    --image_size 256  --map_name uv_seg  --swap_part body  \
    --ip http://10.10.10.100 --port 31102   \
    --has_detector  --bg_ks 7 --ft_ks 3   --front_warp  \
    --bg_model /root/poseGANs/meta_train/background_inpaintor/net_epoch_50_id_G.pth \
    --src_path  /root/poseGANs/meta_train/samples/all_img/men1_256.jpg \
    --tgt_path  /root/poseGANs/meta_train/samples/all_img/8_256.jpg  --visual


--src_path /public/liuwen/p300/results/impersonator_02_14/024_7_2/0000.jpg    \
--tgt_path /public/liuwen/p300/results/impersonator_02_14/023_3_1/000.jpg  --visual



