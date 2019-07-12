#--src_path /public/liuwen/p300/imgs/3.jpg    \
#--src_path /public/liuwen/p300/ImPer/motion_transfer_HD/001/1/1/0000.jpg
#--tgt_path /home/piaozx/liuwen/p300/human_pose/processed/motion_transfer_HD/001/1/2/0640.jpg  --visual \


#--src_path /public/liuwen/p300/imgs/3.jpg    \
#--tgt_path /public/liuwen/p300/ImPer/motion_transfer_HD/024/8/2/0695.jpg  \

#--src_path /public/deep_fashion/intrinsic/img_256/fashionMENShirts_Polosid0000473801_3back.jpg
#--tgt_path /public/deep_fashion/intrinsic/img_256/fashionMENShirts_Polosid0000473801_7additional.jpg

python demo_imitator.py --gpu_ids 9 \
    --model imper_v2_fixbg_imitator \
    --name imper_v2_fixbg \
    --gen_name res_unet_front \
    --output_dir '' \
    --checkpoints_dir /public/liuwen/p300/models/imper \
    --ip http://10.19.129.76 --port 10086   \
    --src_path /public/liuwen/p300/ImPer/motion_transfer_HD/001/1/1/0000.jpg    \
    --tgt_path /public/liuwen/p300/ImPer/motion_transfer_HD/024/8/2/0695.jpg  \
    --visual --has_detector  --bg_ks 9  --load_epoch 20


    --src_path /public/deep_fashion/intrinsic/img_256/fashionMENShirts_Polosid0000473801_3back.jpg \
    --tgt_path /public/deep_fashion/intrinsic/img_256/fashionMENShirts_Polosid0000473801_7additional.jpg \


python demo_imitator.py --gpu_ids 9 \
    --model imper_v2_fixbg_imitator \
    --name intrinsic_mixup_hmr \
    --gen_name res_unet_front \
    --output_dir '' \
    --checkpoints_dir /public/liuwen/p300/models/deep_fashion_hmr_dp_maxbbox \
    --ip http://10.19.129.76 --port 10086   \
    --src_path /public/deep_fashion/intrinsic/img_256/fashionWOMENDressesid0000716201_4full.jpg \
    --tgt_path /public/deep_fashion/intrinsic/img_256/fashionWOMENDressesid0000716201_3back.jpg \
    --visual --has_detector  --bg_ks 25  --load_epoch 33 \
    --cam_strategy copy
