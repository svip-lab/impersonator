# fix bg Imper (HMR process)
python train.py --gpu_ids 0,1,2,3,4,5,6,7 \
    --data_dir  /public/liuwen/p300/ImPer    \
    --place_dir  /public/liuwen/p300/places365_standard     \
    --checkpoints_dir  /public/liuwen/p300/models/imper  \
    --train_ids_file pairs_train.pkl \
    --test_ids_file pairs_val.pkl \
    --model imper_v2_fixbg \
    --gen_name res_unet_front \
    --name imper_v2_fixbg \
    --dataset_mode  deep_fashion --is_both \
    --use_vgg  --use_face \
    --batch_size 48  \
    --lambda_mask 1.0 --lambda_vgg 10.0 --lambda_face 5.0 \
    --train_G_every_n_iterations 5  \
    --nepochs_no_decay 10  --nepochs_decay 10  \
    --bg_ks 13  --ft_ks 5

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

