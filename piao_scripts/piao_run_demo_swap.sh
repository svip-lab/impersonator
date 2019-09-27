

# RTX 01, impersonator v3
python piao_demo_swap.py --gpu_ids 1 \
    --model swapper_v2 \
    --gen_name impersonator \
    --name impersonator_mi_fashion_place \
    --checkpoints_dir /public/liuwen/p300/models \
    --output_dir meta_train/impersonator_mi \
    --image_size 256  --map_name uv_seg  --swap_part body  \
    --ip http://10.10.10.100 --port 8095   \
    --has_detector  --bg_ks 25 --ft_ks 3 \
    --bg_model meta_train/background_inpaintor/net_epoch_50_id_G.pth


#--src_path /public/liuwen/p300/results/impersonator_02_14/024_7_2/0000.jpg    \
#--tgt_path /public/liuwen/p300/results/impersonator_02_14/023_3_1/000.jpg  --visual



