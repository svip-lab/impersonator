 # baselines
 ### Concat
python train.py --gpu_ids 0,1 \
    --data_dir  /public/deep_fashion/intrinsic    \
    --place_dir  /public/liuwen/p300/places365_standard     \
    --checkpoints_dir  /public/liuwen/p300/models   \
    --train_ids_file pairs_train.pkl \
    --test_ids_file pairs_test.pkl \
    --model concat \
    --name df_concat \
    --dataset_mode  deep_fashion \
    --use_vgg  --use_face \
    --batch_size 24  --norm_type instance   \
    --lambda_mask 1.0 --lambda_vgg 10.0 --lambda_face 5.0 \
    --train_G_every_n_iterations 5  \
    --nepochs_no_decay 5  --nepochs_decay 15  \
    --bg_ks 23  --ft_ks 9

 ### Texture
python train.py --gpu_ids 2,3 \
    --data_dir  /public/deep_fashion/intrinsic    \
    --place_dir  /public/liuwen/p300/places365_standard     \
    --checkpoints_dir  /public/liuwen/p300/models   \
    --train_ids_file pairs_train.pkl \
    --test_ids_file pairs_test.pkl \
    --model texture \
    --name df_texture \
    --dataset_mode  deep_fashion \
    --use_vgg  --use_face \
    --batch_size 20  --norm_type instance   \
    --lambda_mask 1.0 --lambda_vgg 10.0 --lambda_face 5.0 \
    --train_G_every_n_iterations 5  \
    --nepochs_no_decay 5  --nepochs_decay 15  \
    --bg_ks 23  --ft_ks 9

 ### Feature
python train.py --gpu_ids 0,1,2,3 \
    --data_dir  /public/deep_fashion/intrinsic    \
    --place_dir  /public/liuwen/p300/places365_standard     \
    --checkpoints_dir  /public/liuwen/p300/models   \
    --train_ids_file pairs_train.pkl \
    --test_ids_file pairs_test.pkl \
    --model feature \
    --name df_feature \
    --dataset_mode  deep_fashion \
    --use_vgg  --use_face \
    --batch_size 32  --norm_type instance   \
    --lambda_mask 1.0 --lambda_vgg 10.0 --lambda_face 5.0 \
    --train_G_every_n_iterations 5  \
    --nepochs_no_decay 5  --nepochs_decay 15  \
    --bg_ks 23  --ft_ks 9