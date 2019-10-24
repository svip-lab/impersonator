#! /bin/bash

# basic configs
#gpu_ids=0,1     # if using multi-gpus, increasing the batch_size
gpu_ids=1

# dataset configs
dataset_model=iPER_place  # use iPER_place dataset
data_dir=/p300/iccv/iPER_examples  # need to be replaced!!!!!
place_dir=/p300/iccv/places365_standard  # need to be replaced!!!!!
images_folder=images
smpls_folder=smpls
train_ids_file=train.txt
test_ids_file=val.txt

# saving configs
checkpoints_dir=/p300/iccv/ckpts_models   # directory to save models, need to be replaced!!!!!
name=exp_iPER_place   # the directory is ${checkpoints_dir}/name, which is used to save the checkpoints.

# model configs
model=impersonator_trainer_aug
gen_name=impersonator
image_size=256

# training configs
load_path="None"
batch_size=4
lambda_rec=10.0
lambda_tsf=10.0
lambda_face=5.0
lambda_style=0.0
lambda_mask=1.0
lambda_mask_smooth=1.0
nepochs_no_decay=5  # fixing learning rate when epoch ranges in [0, 5]
nepochs_decay=25    # decreasing the learning rate when epoch ranges in [6, 25+5]

python train.py --gpu_ids ${gpu_ids}        \
    --data_dir         ${data_dir}          \
    --place_dir        ${place_dir}         \
    --images_folder    ${images_folder}     \
    --smpls_folder     ${smpls_folder}      \
    --checkpoints_dir  ${checkpoints_dir}   \
    --load_path        ${load_path}         \
    --train_ids_file   ${train_ids_file}    \
    --test_ids_file    ${test_ids_file}     \
    --model            ${model}             \
    --gen_name         ${gen_name}          \
    --name             ${name}              \
    --dataset_mode     ${dataset_model}     \
    --image_size       ${image_size}        \
    --batch_size       ${batch_size}        \
    --lambda_face      ${lambda_face}       \
    --lambda_tsf       ${lambda_tsf}        \
    --lambda_style     ${lambda_style}      \
    --lambda_rec       ${lambda_rec}        \
    --lambda_mask      ${lambda_mask}       \
    --lambda_mask_smooth  ${lambda_mask_smooth} \
    --nepochs_no_decay ${nepochs_no_decay}  --nepochs_decay ${nepochs_decay}  \
    --mask_bce     --use_vgg     --use_style     --use_face