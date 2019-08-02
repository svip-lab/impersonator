#python demo.py --gpu_ids 9 \
#    --model imitator \
#    --gen_name impersonator \
#    --dataset_mode  mi_v2 \
#    --name impersonator_02 \
#    --output_dir  '' \
#    --image_size 256  \
#    --map_name uv_seg   \
#    --src_path /public/liuwen/p300/imgs/3.jpg    \
#    --tgt_path /public/liuwen/p300/ImPer/motion_transfer_HD/024/8/2/0695.jpg  --visual \
#    --ip http://10.19.129.76 \
#    --port 10086 \
#    --checkpoints_dir /public/liuwen/p300/models --do_saturate_mask  \
#    --bg_replace


#3_256.jpg  google1.jpg  ins1.jpg  ins3.jpg  ins5.jpg  ins7.jpg  men1_256.jpg
#8_256.jpg  google2.jpg  ins2.jpg  ins4.jpg  ins6.jpg  ins8.jpg  women1_256.jpg

## RTX 01, impersonator v3 (currently works)
python demo_imitator.py --gpu_ids 0 \
    --model imitator_v2 \
    --gen_name impersonator \
    --name impersonator_mi \
    --checkpoints_dir pretrains/models \
    --output_dir meta_train/impersonator_mi  \
    --image_size 256  --map_name uv_seg   \
    --ip http://10.10.10.100 --port  8096 \
    --bg_model meta_train/background_inpaintor/net_epoch_50_id_G.pth \
    --src_path meta_train/samples/all_img/men1_256.jpg    \
    --tgt_path meta_train/samples/ref_imgs/2  \
    --pri_path meta_train/samples/ref_imgs/1  \
    --has_detector  --bg_ks 7 --ft_ks 3  \
    --front_warp \
    --visual

