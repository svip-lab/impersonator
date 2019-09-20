## AI Docker, impersonator in ICCV
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

## RTX 01, impersonator v3 (currently works)
python demo_imitator.py --gpu_ids 0 \
    --model imitator_v2 \
    --gen_name impersonator \
    --name impersonator_mi \
    --checkpoints_dir /public/liuwen/p300/models \
    --output_dir /root/poseGANs/meta_train/impersonator_mi  \
    --image_size 256  --map_name uv_seg   \
    --ip http://10.10.10.100 --port 31102   \
    --bg_model /root/poseGANs/meta_train/background_inpaintor/net_epoch_50_id_G.pth \
    --src_path /root/poseGANs/meta_train/samples/all_img/men1_256.jpg    \
    --tgt_path /root/poseGANs/meta_train/samples/ref_imgs/2  \
    --pri_path /root/poseGANs/meta_train/samples/ref_imgs/1  \
    --has_detector  --bg_ks 7 --ft_ks 3  --front_warp --post_tune --visual


python demo_imitator.py --gpu_ids 0 \
    --model imitator_v2 \
    --gen_name impersonator \
    --name impersonator_mi_fashion_place \
    --checkpoints_dir /public/liuwen/p300/models \
    --output_dir /root/poseGANs/meta_train/impersonator_mi  \
    --image_size 256  --map_name uv_seg   \
    --ip http://10.10.10.100 --port 31102   \
    --bg_model /root/poseGANs/meta_train/background_inpaintor/net_epoch_50_id_G.pth \
    --src_path /root/poseGANs/meta_train/samples/all_img/fashionWOMENDressesid0000271801_4full.jpg    \
    --tgt_path /root/poseGANs/meta_train/samples/ref_imgs/2  \
    --pri_path /root/poseGANs/meta_train/samples/ref_imgs/1  \
    --has_detector  --bg_ks 25 --ft_ks 3  --front_warp --post_tune --visual


3_256.jpg  men1_256.jpg 8_256.jpg  women1_256.jpg  fashionWOMENDressesid0000271801_4full.jpg

python demo_imitator.py --gpu_ids 0 \
    --model imitator_v2 \
    --gen_name impersonator \
    --name intrinsic_mixup_hmr \
    --checkpoints_dir /public/liuwen/p300/models/deep_fashion_hmr_dp_maxbbox \
    --load_epoch 20  \
    xxxxxx
