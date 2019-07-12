# do not fix bg, df + imper
python train.py --gpu_ids 0,1,2,3,4,5,6,7 \
    --data_dir  /public/deep_fashion/intrinsic    \
    --train_ids_file pairs_train.pkl \
    --test_ids_file pairs_test.pkl \
    --images_folder img_256 \
    --train_pkl_folder  train_256_v2_max_bbox_dp_hmr_pairs_results \
    --test_pkl_folder test_256_v2_max_bbox_dp_hmr_pairs_results \
    --place_dir  /public/liuwen/p300/places365_standard     \
    --checkpoints_dir  /public/liuwen/p300/models   \
    --model imper_v2 \
    --name imper_v2_df_imper \
    --gen_name res_share_unet \
    --dataset_mode  df_imper \
    --use_vgg  --use_face \
    --batch_size 48  --norm_type instance   \
    --lambda_mask 0.1 --lambda_vgg 10.0 --lambda_face 5.0 \
    --D_layers 4 --final_lr 0.00001 \
    --train_G_every_n_iterations 5  \
    --nepochs_no_decay 30  --nepochs_decay 0  \
    --bg_ks 23  --ft_ks 9


# fix bg Imper
python train.py --gpu_ids 0,3,4,9 \
    --data_dir  /public/liuwen/p300/ImPer    \
    --place_dir  /public/liuwen/p300/places365_standard     \
    --checkpoints_dir  /public/liuwen/p300/models   \
    --train_ids_file pairs_train_divide_idx_15.pkl \
    --test_ids_file pairs_val_divide_idx_15.pkl \
    --model imper_v2_fixbg \
    --name imper_v2_fixbg_imper \
    --gen_name res_unet_front \
    --dataset_mode  imper_pair --is_both \
    --use_vgg  --use_face \
    --batch_size 24   \
    --lambda_mask 0.5 \
    --train_G_every_n_iterations 5  \
    --nepochs_no_decay 2  --nepochs_decay 18 \
    --bg_ks 9  --ft_ks 3


python train.py --gpu_ids 0 \
    --data_dir  /public/liuwen/p300/ImPer    \
    --place_dir  /public/liuwen/p300/places365_standard     \
    --checkpoints_dir  /public/liuwen/p300/models   \
    --train_ids_file pairs_train_divide_idx_15.pkl \
    --test_ids_file pairs_val_divide_idx_15.pkl \
    --model imper_v2_fixbg \
    --gen_name res_plain_unet_front \
    --name imper_v2_plain_fixbg_imper \
    --dataset_mode  imper_pair --is_both \
    --use_vgg  --use_face \
    --batch_size 8   \
    --lambda_mask 1.0 --lambda_lp 5.0 --lambda_vgg 5.0 \
    --train_G_every_n_iterations 1  \
    --nepochs_no_decay 5  --nepochs_decay 15


python train.py --gpu_ids 9 \
    --data_dir  /public/liuwen/p300/ImPer    \
    --place_dir  /public/liuwen/p300/places365_standard     \
    --checkpoints_dir  /public/liuwen/p300/models   \
    --train_ids_file pairs_train_divide_idx_15.pkl \
    --test_ids_file pairs_val_divide_idx_15.pkl \
    --model imper_v2_fixbg \
    --gen_name res_share_unet \
    --name share_imper_v2_fixbg_imper \
    --dataset_mode  imper_pair --is_both \
    --use_vgg  --use_face \
    --batch_size 8   \
    --lambda_mask 1.0 \
    --train_G_every_n_iterations 2  \
    --nepochs_no_decay 5  --nepochs_decay 15

# dot not fix bg Imper
python train.py --gpu_ids 9 \
    --data_dir  /public/liuwen/p300/ImPer    \
    --place_dir  /public/liuwen/p300/places365_standard     \
    --checkpoints_dir  /public/liuwen/p300/models   \
    --train_ids_file pairs_train_divide_idx_15.pkl \
    --test_ids_file pairs_val_divide_idx_15.pkl \
    --model imper_v2 \
    --name imper_v2_imper \
    --dataset_mode  imper_pair  \
    --use_vgg  --use_face \
    --batch_size 4   \
    --lambda_mask 1.0 \
    --train_G_every_n_iterations 5  \
    --nepochs_no_decay 5  --nepochs_decay 15


# fix bg DP
python train.py --gpu_ids 5,6,7,8 \
    --data_dir  /public/liuwen/p300/deep_fashion    \
    --place_dir  /public/liuwen/p300/places365_standard     \
    --checkpoints_dir  /public/liuwen/p300/models   \
    --train_ids_file pairs_train.pkl \
    --test_ids_file pairs_test.pkl \
    --model imper_v2_fixbg \
    --gen_name res_unet_front \
    --name imper_v2_fixbg_dp_maxbbox \
    --dataset_mode  deep_fashion --is_both \
    --use_vgg  --use_face \
    --batch_size 24   \
    --lambda_mask 1.0 \
    --train_G_every_n_iterations 1  \
    --nepochs_no_decay 5  --nepochs_decay 15  \
    --bg_ks 21  --ft_ks 9

python train.py --gpu_ids 0,1,2,3 \
    --data_dir  /public/liuwen/p300/deep_fashion    \
    --place_dir  /public/liuwen/p300/places365_standard     \
    --checkpoints_dir  /public/liuwen/p300/models   \
    --train_ids_file pairs_train.pkl \
    --test_ids_file pairs_test.pkl \
    --model imper_v2_fixbg \
    --gen_name res_unet_front \
    --name imper_v2_fixbg_dp_maxbbox_v2 \
    --dataset_mode  deep_fashion --is_both \
    --use_vgg  --use_face \
    --batch_size 24   --D_layers 3 \
    --lambda_mask 1.0 --lambda_vgg 100.0 --lambda_face 50.0 \
    --train_G_every_n_iterations 5  \
    --nepochs_no_decay 5  --nepochs_decay 15  \
    --bg_ks 21  --ft_ks 9


python train.py --gpu_ids 4,9 \
    --data_dir  /public/liuwen/p300/deep_fashion    \
    --place_dir  /public/liuwen/p300/places365_standard     \
    --checkpoints_dir  /public/liuwen/p300/models   \
    --train_ids_file pairs_train.pkl \
    --test_ids_file pairs_test.pkl \
    --model imper_v2_fixbg \
    --gen_name res_unet_front \
    --name imper_v2_fixbg_dp_maxbbox_v3 \
    --dataset_mode  deep_fashion --is_both \
    --use_vgg  --use_face \
    --batch_size 16   --D_layers 4 --norm_type batch \
    --lambda_mask 1.0 --lambda_vgg 100.0 --lambda_face 50.0 \
    --train_G_every_n_iterations 5  \
    --nepochs_no_decay 5  --nepochs_decay 15  \
    --bg_ks 21  --ft_ks 9


 # fix bg HMR
python train.py --gpu_ids 0,1,2,3 \
    --data_dir  /public/deep_fashion/intrinsic    \
    --place_dir  /public/liuwen/p300/places365_standard     \
    --checkpoints_dir  /public/liuwen/p300/models   \
    --train_ids_file pairs_train.pkl \
    --test_ids_file pairs_test.pkl \
    --model imper_v2_fixbg \
    --gen_name res_unet_front \
    --name imper_v2_fixbg_df_hmr \
    --dataset_mode  deep_fashion_hmr --is_both \
    --use_vgg  --use_face \
    --batch_size 24  --norm_type batch   \
    --lambda_mask 1.0 --lambda_vgg 10.0 --lambda_face 5.0 \
    --train_G_every_n_iterations 5  \
    --nepochs_no_decay 5  --nepochs_decay 15  \
    --bg_ks 23  --ft_ks 9


python train.py --gpu_ids 1,2 \
    --data_dir  /public/liuwen/p300/deep_fashion    \
    --place_dir  /public/liuwen/p300/places365_standard     \
    --checkpoints_dir  /public/liuwen/p300/models   \
    --train_ids_file pairs_train.pkl \
    --test_ids_file pairs_test.pkl \
    --model imper_v2_fixbg \
    --gen_name res_unet_front \
    --name imper_v2_fixbg_dp_maxbbox_lp2.0_vgg2.0 \
    --dataset_mode  deep_fashion --is_both \
    --use_vgg  --use_face \
    --batch_size 10   \
    --lambda_mask 0.1  --lambda_lp 2.0 --lambda_vgg 2.0 \
    --train_G_every_n_iterations 1  \
    --nepochs_no_decay 2  --nepochs_decay 18  \
    --bg_ks 21  --ft_ks 9