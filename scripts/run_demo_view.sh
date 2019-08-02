# RTX 01, impersonator v3
python demo_view.py --gpu_ids 9 \
    --model viewer_v2 \
    --gen_name impersonator \
    --name impersonator_mi \
    --checkpoints_dir /public/liuwen/p300/models \
    --output_dir /root/poseGANs/meta_train/impersonator_mi \
    --image_size 256  --map_name uv_seg  --swap_part body  \
    --ip http://10.10.10.100 --port 31102   \
    --has_detector  --bg_ks 7 --ft_ks 3   --front_warp --post_tune \
    --bg_model /root/poseGANs/meta_train/background_inpaintor/net_epoch_50_id_G.pth \
    --src_path  /root/poseGANs/meta_train/samples/all_img/men1_256.jpg \
    --tgt_path  /root/poseGANs/meta_train/samples/all_img/8_256.jpg  --visual