#--src_path /public/liuwen/p300/imgs/3.jpg    \
#--src_path /public/liuwen/p300/ImPer/motion_transfer_HD/001/1/1/0000.jpg
#--tgt_path /home/piaozx/liuwen/p300/human_pose/processed/motion_transfer_HD/001/1/2/0640.jpg  --visual \


#--src_path /public/liuwen/p300/imgs/3.jpg    \
#--tgt_path /public/liuwen/p300/ImPer/motion_transfer_HD/024/8/2/0695.jpg  \

#--src_path /public/deep_fashion/intrinsic/img_256/fashionMENShirts_Polosid0000473801_3back.jpg
#--tgt_path /public/deep_fashion/intrinsic/img_256/fashionMENShirts_Polosid0000473801_7additional.jpg

####################################### W_lbw ##########################################
python evaluation/eval_df_im.py --gpu_ids 2 \
    --model imper_v2_fixbg \
    --name market1501_mixup_hmr_bn \
    --gen_name res_unet_front \
    --data_dir  /p300_piaozx/market_data  \
    --train_ids_file  eval_pairs_test.pkl \
    --test_ids_file   eval_unpairs_test.pkl \
    --train_pkl_folder pair_v2_show_pairs_results \
    --test_pkl_folder unpair_v2_show_pairs_results \
    --images_folder test_256 \
    --dataset_mode  deep_fashion --is_both \
    --bg_model /p300_piaozx/impersonator_v2_checkpoints/background_inpaintor/net_epoch_50_id_G.pth \
    --checkpoints_dir /p300_piaozx/impersonator_v2_checkpoints \
    --out_dir   /p300/poseGANs/comps/market/W_lfw \
    --ip http://10.10.10.100 --port 10087   \
    --bg_ks 23  --ft_ks 9 --load_epoch 8   \
    --serial_batches \
    --visual


python evaluation/eval_df_im.py --gpu_ids 3 \
    --model imper_v2_fixbg \
    --name market1501_mixup_hmr_real_bn \
    --gen_name res_unet_front \
    --data_dir  /p300_piaozx/market_data  \
    --train_ids_file  eval_pairs_test.pkl \
    --test_ids_file   eval_unpairs_test.pkl \
    --train_pkl_folder pair_v2_show_pairs_results \
    --test_pkl_folder unpair_v2_show_pairs_results \
    --images_folder test_256 --norm_type batch  --batch_size 32 \
    --dataset_mode  deep_fashion --is_both \
    --bg_model /p300_piaozx/impersonator_v2_checkpoints/background_inpaintor/net_epoch_50_id_G.pth \
    --checkpoints_dir /p300_piaozx/impersonator_v2_checkpoints \
    --out_dir   /p300/poseGANs/comps/market/W_lfw_bn \
    --ip http://10.10.10.100 --port 10087   \
    --bg_ks 23  --ft_ks 9 --load_epoch 7   \
    --serial_batches \
    --visual

####################################### W_c ##########################################
python evaluation/eval_df_im.py --gpu_ids 0 \
    --model concat \
    --name df_concat \
    --data_dir  /p300_piaozx/market_data  \
    --train_ids_file  eval_pairs_test.pkl \
    --test_ids_file   eval_unpairs_test.pkl \
    --train_pkl_folder pair_v2_show_pairs_results \
    --test_pkl_folder unpair_v2_show_pairs_results \
    --images_folder test_256 \
    --dataset_mode  deep_fashion \
    --checkpoints_dir /p300/poseGANs/models/market \
    --out_dir   /p300/poseGANs/comps/market/W_c \
    --ip http://10.10.10.100 --port 10087   \
    --bg_ks 23  --ft_ks 9 --load_epoch 4   \
    --serial_batches \
    --visual


python evaluation/eval_df_im.py --gpu_ids 1 \
    --model concat \
    --name df_concat \
    --data_dir  /p300_piaozx/market_data  \
    --train_ids_file  eval_pairs_test.pkl \
    --test_ids_file   eval_unpairs_test.pkl \
    --train_pkl_folder pair_v2_show_pairs_results \
    --test_pkl_folder unpair_v2_show_pairs_results \
    --images_folder test_256 \
    --dataset_mode  deep_fashion \
    --checkpoints_dir /p300/poseGANs/models/market \
    --out_dir   /p300/poseGANs/comps/market/pG2 \
    --ip http://10.10.10.100 --port 10087   \
    --bg_ks 23  --ft_ks 9 --load_epoch 2   \
    --serial_batches


####################################### W_t ##########################################
python evaluation/eval_df_im.py --gpu_ids 2 \
    --model texture \
    --name df_texture \
    --data_dir  /p300_piaozx/market_data  \
    --train_ids_file  eval_pairs_test.pkl \
    --test_ids_file   eval_unpairs_test.pkl \
    --train_pkl_folder pair_v2_show_pairs_results \
    --test_pkl_folder unpair_v2_show_pairs_results \
    --images_folder test_256 \
    --dataset_mode  deep_fashion \
    --checkpoints_dir /p300/poseGANs/models/market \
    --out_dir   /p300/poseGANs/comps/market/W_t \
    --ip http://10.10.10.100 --port 10087   \
    --bg_ks 23  --ft_ks 9 --load_epoch 4   \
    --serial_batches \
    --visual

python evaluation/eval_df_im.py --gpu_ids 3 \
    --model texture \
    --name df_texture \
    --data_dir  /p300_piaozx/market_data  \
    --train_ids_file  eval_pairs_test.pkl \
    --test_ids_file   eval_unpairs_test.pkl \
    --train_pkl_folder pair_v2_show_pairs_results \
    --test_pkl_folder unpair_v2_show_pairs_results \
    --images_folder test_256 \
    --dataset_mode  deep_fashion \
    --checkpoints_dir /p300/poseGANs/models/market \
    --out_dir   /p300/poseGANs/comps/market/SHUP \
    --ip http://10.10.10.100 --port 10087   \
    --bg_ks 23  --ft_ks 9 --load_epoch 2   \
    --serial_batches \
    --visual


####################################### W_f ##########################################
python evaluation/eval_df_im.py --gpu_ids 5 \
    --model feature \
    --name df_feature \
    --data_dir  /p300_piaozx/market_data  \
    --train_ids_file  eval_pairs_test.pkl \
    --test_ids_file   eval_unpairs_test.pkl \
    --train_pkl_folder pair_v2_show_pairs_results \
    --test_pkl_folder unpair_v2_show_pairs_results \
    --images_folder test_256 \
    --dataset_mode  deep_fashion \
    --checkpoints_dir /p300/poseGANs/models/market \
    --out_dir   /p300/poseGANs/comps/market/W_f \
    --ip http://10.10.10.100 --port 10087   \
    --bg_ks 23  --ft_ks 9 --load_epoch 2   \
    --serial_batches \
    --visual


####################################### pG2 ##########################################
python evaluation/eval_df_im.py --gpu_ids 9 \
    --model concat \
    --name df_concat \
    --data_dir  /public/deep_fashion/intrinsic  \
    --train_ids_file  eval_pairs_test.pkl \
    --test_ids_file   eval_unpairs_test.pkl \
    --train_pkl_folder pair_v2_show_pairs_results \
    --test_pkl_folder unpair_v2_show_pairs_results \
    --images_folder img_256 \
    --dataset_mode  deep_fashion \
    --checkpoints_dir /public/liuwen/p300/models \
    --out_dir   /public/liuwen/p300/results/comps/DeepFashion/pG2 \
    --ip http://10.19.129.76 --port 10086   \
    --bg_ks 23  --ft_ks 9 --load_epoch 8   \
    --serial_batches


python demo_imitator.py --gpu_ids 9 \
    --model imper_v2_fixbg \
    --name intrinsic_mixup_hmr \
    --gen_name res_unet_front \
    --output_dir '' \
    --checkpoints_dir /public/liuwen/p300/models/deep_fashion_hmr_dp_maxbbox \
    --ip http://10.19.129.76 --port 10086   \
    --src_path /public/deep_fashion/intrinsic/img_256/fashionWOMENDressesid0000716201_4full.jpg \
    --tgt_path /public/deep_fashion/intrinsic/img_256/fashionWOMENDressesid0000716201_3back.jpg \
    --visual --has_detector  --bg_ks 25  --load_epoch 33 \
    --cam_strategy copy
