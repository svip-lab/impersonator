#!/usr/bin/env bash
############# baselines ##############
######## 1. concat
python train.py --gpu_ids 1 \
    --model concat \
    --dataset_mode  imper \
    --name concat \
    --image_size 256 \
    --map_name uv_seg \
    --batch_size 8  --use_vgg  --use_face \
    --lambda_face 5.0  \
    --train_G_every_n_iterations 5  \
    --nepochs_no_decay 2  --nepochs_decay 28

######## 2. texture warping
python train.py --gpu_ids 3 \
    --model texture_warping \
    --name texture_warping \
    --dataset_mode  imper \
    --map_name uv_seg \
    --image_size 256 \
    --batch_size 16  --use_vgg  --use_face \
    --lambda_face 5.0  \
    --train_G_every_n_iterations 5  \
    --nepochs_no_decay 2  --nepochs_decay 28

######## 3. feature warping
python train.py --gpu_ids 0 \
    --model feature_warping \
    --name feature_warping \
    --gen_name impersonator \
    --dataset_mode  imper \
    --map_name uv_seg \
    --image_size 256  \
    --batch_size 8  --use_vgg  --use_face \
    --lambda_face 5.0  \
    --train_G_every_n_iterations 5  \
    --nepochs_no_decay 2  --nepochs_decay 28

python train.py --gpu_ids 0,1 \
    --model impersonator_02 \
    --name impersonator_02 \
    --gen_name impersonator \
    --dataset_mode  imper \
    --image_size 256  \
    --map_name uv_seg \
    --batch_size 4  --use_vgg  --use_face \
    --lambda_face 5.0 \
    --train_G_every_n_iterations 5

###### impersonator trainer, face 2.5, mask_smooth 0.0
python train.py --gpu_ids 0,1,2,3,4 \
    --data_dir  /public/liuwen/p300/ImPer \
    --place_dir  /public/liuwen/p300/places365_standard \
    --checkpoints_dir  /public/liuwen/p300/models \
    --model impersonator_trainer \
    --gen_name impersonator \
    --name impersonator \
    --dataset_mode  imper \
    --image_size 256  \
    --map_name uv_seg \
    --batch_size 24  --use_vgg  --use_face \
    --lambda_face 2.5 --mask_bce --lambda_mask 1.0 --lambda_mask_smooth 0.0 \
    --nepochs_no_decay 5  --nepochs_decay 20

###### impersonator trainer, face 5.0, mask_smooth 1.0
python train.py --gpu_ids 5,6,7,8,9 \
    --data_dir  /public/liuwen/p300/ImPer \
    --place_dir  /public/liuwen/p300/places365_standard \
    --checkpoints_dir  /public/liuwen/p300/models \
    --model impersonator_trainer \
    --gen_name impersonator \
    --name impersonator_mi \
    --dataset_mode  mi \
    --image_size 256  \
    --map_name uv_seg \
    --batch_size 24  --use_vgg  --use_face \
    --lambda_face 5.0 --mask_bce --lambda_mask 1.0 --lambda_mask_smooth 1.0 \
    --nepochs_no_decay 5  --nepochs_decay 20

###### impersonator trainer, face 5.0, mask_smooth 1.0
python train.py --gpu_ids 0,1,2,3 \
    --data_dir  /public/liuwen/p300/ImPer \
    --place_dir  /public/liuwen/p300/places365_standard \
    --checkpoints_dir  /public/liuwen/p300/models \
    --model impersonator_trainer \
    --gen_name impersonator \
    --name impersonator_imper \
    --dataset_mode  imper \
    --image_size 256  \
    --map_name uv_seg \
    --batch_size 20  --use_vgg  --use_face \
    --lambda_face 5.0 --mask_bce --lambda_mask 1.0 --lambda_mask_smooth 1.0 \
    --nepochs_no_decay 2  --nepochs_decay 28

python train.py --gpu_ids 0,1,2,3,4 \
    --data_dir  /public/liuwen/p300/ImPer \
    --place_dir  /public/liuwen/p300/places365_standard \
    --checkpoints_dir  /public/liuwen/p300/models \
    --model impersonator_trainer \
    --gen_name impersonator \
    --name impersonator_imper_style \
    --dataset_mode  imper \
    --image_size 256  \
    --map_name uv_seg \
    --batch_size 24  --use_vgg --use_style --use_face \
    --lambda_face 5.0 --lambda_vgg 5.0 --lambda_style 5.0 --lambda_lp 5.0  --mask_bce --lambda_mask 1.0 --lambda_mask_smooth 1.0 \
    --nepochs_no_decay 5  --nepochs_decay 25

python train.py --gpu_ids 5,6,7,8,9 \
    --data_dir  /public/liuwen/p300/ImPer \
    --place_dir  /public/liuwen/p300/places365_standard \
    --checkpoints_dir  /public/liuwen/p300/models \
    --model impersonator_trainer_aug \
    --gen_name impersonator \
    --name impersonator_imper_place \
    --dataset_mode  miv2_place --intervals 4 \
    --image_size 256  \
    --map_name uv_seg \
    --batch_size 20  --use_vgg --use_style --use_face \
    --lambda_face 5.0 --lambda_vgg 5.0 --lambda_style 5.0 --lambda_lp 5.0  --mask_bce --lambda_mask 1.0 --lambda_mask_smooth 1.0 \
    --nepochs_no_decay 20  --nepochs_decay 10


python train.py --gpu_ids 5,6,7,8,9 \
    --data_dir  /public/liuwen/p300/ImPer \
    --place_dir  /public/liuwen/p300/places365_standard \
    --fashion_dir /public/deep_fashion/intrinsic \
    --checkpoints_dir  /public/liuwen/p300/models \
    --model impersonator_all_set_trainer_aug \
    --gen_name impersonator \
    --name impersonator_mi_fashion_place \
    --dataset_mode  miv2_fashion_place --intervals 4 \
    --image_size 256  \
    --map_name uv_seg \
    --batch_size 15  --use_vgg --use_style --use_face \
    --lambda_face 5.0 --lambda_vgg 5.0 --lambda_style 5.0 --lambda_lp 5.0  --mask_bce --lambda_mask 1.0 --lambda_mask_smooth 1.0 \
    --nepochs_no_decay 20  --nepochs_decay 10  --debug


python train.py --gpu_ids 4,5,6,7 \
    --data_dir  /public/liuwen/p300/ImPer \
    --place_dir  /public/liuwen/p300/places365_standard \
    --checkpoints_dir  /public/liuwen/p300/models \
    --model impersonator_trainer \
    --gen_name impersonator \
    --name impersonator_512 \
    --dataset_mode  imper \
    --image_size 512  \
    --map_name uv_seg \
    --batch_size 8  --use_vgg  --use_face \
    --lambda_face 5.0 --mask_bce --lambda_mask 1.0 --lambda_mask_smooth 1.0 \
    --nepochs_no_decay 2  --nepochs_decay 28

###### impersonator trainer aug
python train.py --gpu_ids 5,6,7,8,9 \
    --data_dir  /public/liuwen/p300/human_pose/processed \
    --place_dir  /public/liuwen/p300/places365_standard \
    --checkpoints_dir  /public/liuwen/p300/models \
    --gen_name impersonator_full_aug \
    --model impersonator_trainer_aug \
    --name impersonator_full_aug \
    --dataset_mode  miv2_place \
    --repeat_num 9 \
    --image_size 256 \
    --map_name uv_seg \
    --batch_size 20  --use_vgg  --use_face \
    --lambda_face 5.0  --lambda_mask 0.5 \
    --train_G_every_n_iterations 5  \
    --nepochs_no_decay 10  --nepochs_decay 40 \
    --display_freq_s 500
