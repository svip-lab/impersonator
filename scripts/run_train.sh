############# baselines ##############
######## 1. concat
python train.py --gpu_ids 1 \
    --model concat \
    --dataset_mode  mi_v2 \
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
    --dataset_mode  mi_v2 \
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
    --dataset_mode  mi_v2 \
    --map_name uv_seg \
    --image_size 256 \
    --batch_size 8  --use_vgg  --use_face \
    --lambda_face 5.0  \
    --train_G_every_n_iterations 5  \
    --nepochs_no_decay 2  --nepochs_decay 28


python train.py --gpu_ids 1,2 \
    --model impersonator \
    --dataset_mode  mi_v2 \
    --name impersonator_01 \
    --image_size 256 \
    --map_name uv_seg \
    --batch_size 8  --use_vgg  --use_face \
    --lambda_face 5.0 \
    --train_G_every_n_iterations 5

python train.py --gpu_ids 0,1 \
    --model impersonator_02 \
    --gen_name impersonator \
    --name impersonator_02 \
    --dataset_mode  mi_v2 \
    --image_size 256 \
    --map_name uv_seg \
    --batch_size 4  --use_vgg  --use_face \
    --lambda_face 5.0 \
    --train_G_every_n_iterations 5

python train.py --gpu_ids 1,2 \
    --model impersonator_02 \
    --gen_name impersonator_full \
    --name impersonator_04 \
    --dataset_mode  fast_mi \
    --image_size 256 \
    --map_name uv_seg \
    --batch_size 12  --use_vgg  --use_face \
    --lambda_face 5.0  \
    --train_G_every_n_iterations 5  \
    --nepochs_no_decay 10  --nepochs_decay 40

python train.py --gpu_ids 2 \
    --model impersonator_03 \
    --gen_name impersonator_adain \
    --name impersonator_03 \
    --dataset_mode  mi_v2 \
    --image_size 256 \
    --map_name uv_seg \
    --batch_size 12  --use_vgg  --use_face \
    --lambda_face 5.0  --lambda_mask 0.5 \
    --train_G_every_n_iterations 5  \
    --nepochs_no_decay 10  --nepochs_decay 40

python train.py --gpu_ids 3 \
    --model impersonator_03 \
    --gen_name impersonator_adain_warp \
    --name impersonator_03_adain \
    --dataset_mode  mi_v2 \
    --repeat_num 9 \
    --image_size 256 \
    --map_name uv_seg \
    --batch_size 4  --use_vgg  --use_face \
    --lambda_face 5.0  --lambda_mask 0.5 \
    --train_G_every_n_iterations 5  \
    --nepochs_no_decay 10  --nepochs_decay 40

python train.py --gpu_ids 3 \
    --model impersonator_03 \
    --gen_name impersonator_adain_warp_v2 \
    --name impersonator_03_adain_v2 \
    --dataset_mode  fast_mi \
    --repeat_num 9 \
    --image_size 256 \
    --map_name uv_seg \
    --batch_size 10  --use_vgg  --use_face \
    --lambda_face 5.0  --lambda_mask 0.5 \
    --train_G_every_n_iterations 5  \
    --nepochs_no_decay 2  --nepochs_decay 30

###### impersonator trainer,  ti
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

###### impersonator trainer,  ai
python train.py --gpu_ids 0,1,2,3,4 \
    --data_dir  /public/liuwen/p300/human_pose/processed \
    --place_dir  /public/liuwen/p300/places365_standard \
    --checkpoints_dir  /public/liuwen/p300/models \
    --gen_name impersonator_full \
    --model impersonator_trainer \
    --name impersonator_full \
    --dataset_mode  mi_v2 \
    --repeat_num 9 \
    --image_size 256 \
    --map_name uv_seg \
    --batch_size 20  --use_vgg  --use_face \
    --lambda_face 5.0  --lambda_mask 0.5 \
    --train_G_every_n_iterations 5  \
    --nepochs_no_decay 10  --nepochs_decay 40