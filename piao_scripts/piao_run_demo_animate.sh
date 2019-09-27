python piao_demo_animate.py --gpu_ids 2 \
    --model animator \
    --gen_name impersonator \
    --name impersonator_mi_fashion_place \
    --output_dir '' \
    --checkpoints_dir /public/liuwen/p300/models \
    --bg_model meta_train/background_inpaintor/net_epoch_50_id_G.pth \
    --image_size 256  \
    --map_name uv_seg   \
    --swap_part body \
    --has_detector  --bg_ks 25 --ft_ks 3 \
    --load_epoch 20


