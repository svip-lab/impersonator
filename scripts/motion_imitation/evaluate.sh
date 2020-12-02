#! /bin/bash

data_dir="/p300/tpami/iPER"  # need to be replaced!!!!!
output_dir="/p300/iccv2019/baselines/LWG-ADD"  # need to be replaced!!!!!

##
gpu=0
name="imper_results"
checkpoints_dir="./outputs/checkpoints/"
## if use ImPer dataset trained model
#load_path="./outputs/checkpoints/lwb_imper/net_epoch_30_id_G.pth"

## if use ImPer and Place datasets trained model
load_path="./outputs/checkpoints/lwb_imper_place/net_epoch_30_id_G.pth"

## if use ImPer, DeepFashion, and Place datasets trained model
#load_path="./outputs/checkpoints/lwb_imper_fashion_place/net_epoch_30_id_G.pth"

## if use DeepFillv2 trained background inpainting network,
bg_model="./outputs/checkpoints/deepfillv2/net_epoch_50_id_G.pth"
## otherwise, it will use the BGNet in the original LiquidWarping GAN
#bg_model="ORIGINAL"


python evaluate.py --gpu_ids ${gpu} \
    --model imitator \
    --gen_name impersonator \
    --image_size 256 \
    --name ${name}  \
    --data_dir  ${data_dir}  \
    --checkpoints_dir ${checkpoints_dir} \
    --bg_model ${bg_model}      \
    --load_path ${load_path}    \
    --output_dir ${output_dir}  \
    --bg_ks 13 --ft_ks 3        \
    --has_detector --post_tune

#python evaluate.py --gpu_ids ${gpu} \
#    --model imitator \
#    --gen_name impersonator \
#    --image_size 256 \
#    --name ${name}  \
#    --data_dir  ${data_dir}  \
#    --checkpoints_dir ${checkpoints_dir} \
#    --bg_model ${bg_model}      \
#    --load_path ${load_path}    \
#    --output_dir ${output_dir}  \
#    --bg_ks 11 --ft_ks 3        \
#    --has_detector

# --front_warp

