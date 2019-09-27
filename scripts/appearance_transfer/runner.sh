#! /bin/bash

# choose other inputs src img and reference images
src_path="./assets/src_imgs/imper_A_Pose/024_8_2_0000.jpg"
tgt_path="./assets/src_imgs/imper_A_Pose/010_2_1_000.jpg"


##
gpu=0
gen_name="impersonator"
name="imper_results"
checkpoints_dir="./outputs/checkpoints/"
output_dir="./outputs/results/"

## if use ImPer dataset trained model
#load_path="./outputs/checkpoints/lwb_imper/net_epoch_30_id_G.pth"

## if use ImPer and Place datasets trained model
#load_path="./outputs/checkpoints/lwb_imper_place/net_epoch_30_id_G.pth"

## if use ImPer, DeepFashion, and Place datasets trained model
load_path="./outputs/checkpoints/lwb_imper_fashion_place/net_epoch_30_id_G.pth"

## if use DeepFillv2 trained background inpainting network,
bg_model="./outputs/checkpoints/deepfillv2/net_epoch_50_id_G.pth"
## otherwise, it will use the BGNet in the original LiquidWarping GAN
#bg_model="ORIGINAL"

python run_swap.py --gpu_ids ${gpu} \
    --model swapper \
    --gen_name impersonator \
    --image_size 256 \
    --swap_part body \
    --name ${name}   \
    --checkpoints_dir ${checkpoints_dir} \
    --bg_model ${bg_model}      \
    --load_path ${load_path}    \
    --output_dir ${output_dir}  \
    --src_path   ${src_path}    \
    --tgt_path   ${tgt_path}    \
    --bg_ks 7 --ft_ks 3         \
    --has_detector  --post_tune  --front_warp --save_res  \
    --ip http://10.10.10.100 --port 31102
