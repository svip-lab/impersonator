#! /bin/bash


# basic configs
gpu_ids=0,1
image_size=256


# model configs
model=impersonator_trainer
gen_name=impersonator


# dataset configs
data_dir=/public/liuwen/p300/iPER  # need to be specified!!!!!
dataset_model=iPER  # use iPER dataset


# saving configs
checkpoints_dir=/public/liuwen/p300/ckpts_models   # directory to save models
name=exp_iPER   # the directory is ${checkpoints_dir}/name, which is used to save the checkpoints.

# training configs
batch_size=8
lambda_face=5.0
lambda_vgg=5.0
lambda_style=5.0
lambda_lp=5.0
lambda_mask=1.0
lambda_mask_smooth=1.0
nepochs_no_decay=5  # fixing learning rate when epoch ranges in [0, 5]
nepochs_decay=25    # decreasing the learning rate when epoch rangs in [6, 25+5]

python train.py --gpu_ids ${gpu_ids}        \
    --data_dir  ${data_dir}                 \
    --checkpoints_dir  ${checkpoints_dir}   \
    --model            ${model}             \
    --gen_name         ${gen_name}          \
    --name             ${name}              \
    --dataset_mode     ${dataset_model}     \
    --image_size       ${image_size}        \
    --batch_size       ${batch_size}        \
    --lambda_face      ${lambda_face}       \
    --lambda_vgg       ${lambda_vgg}        \
    --lambda_style     ${lambda_style}      \
    --lambda_lp        ${lambda_lp}         \
    --lambda_mask      ${lambda_mask}       \
    --lambda_mask_smooth  ${lambda_mask_smooth} \
    --nepochs_no_decay ${nepochs_no_decay}  --nepochs_decay ${nepochs_decay}  \
    --mask_bce     --use_vgg     --use_style     --use_face

