############# baselines ##############
######## 1. concat
python evaluate.py --gpu_ids 0 \
    --model concat \
    --gen_name concat \
    --dataset_mode  mi_v2 \
    --name concat \
    --output_dir /p300/poseGANs \
    --image_size 256  \
    --map_name uv_seg  --load_epoch 20

######## 2. texture_warping
python evaluate.py --gpu_ids 7 \
    --model texture_warping \
    --gen_name concat \
    --name texture_warping \
    --dataset_mode  mi_v2 \
    --output_dir /p300/poseGANs \
    --image_size 256  \
    --map_name uv_seg  --load_epoch 10

######## 3. feature_warping
python evaluate.py --gpu_ids 3 \
    --model feature_warping \
    --gen_name impersonator \
    --name feature_warping \
    --dataset_mode  mi_v2 \
    --output_dir /p300/poseGANs \
    --image_size 256  \
    --map_name uv_seg  --load_epoch 2


python evaluate.py --gpu_ids 2 \
    --model impersonator_01 \
    --dataset_mode  fast_mi \
    --name impersonator_01 \
    --image_size 256  \
    --map_name uv_seg


# 6 epoch
python evaluate.py --gpu_ids 1 \
    --model imitator \
    --gen_name impersonator \
    --dataset_mode  mi_v2 \
    --name impersonator_02 \
    --output_dir /p300/poseGANs \
    --image_size 256  \
    --map_name uv_seg --cam_strategy smooth


##### ours, AdaIN-Warping, impersonator_03_adain, impersonator_adain_warp
python evaluate.py --gpu_ids 0 \
    --model imitator \
    --gen_name impersonator_adain_warp \
    --name impersonator_03_adain \
    --repeat_num 9 \
    --output_dir /p300/poseGANs \
    --image_size 256  \
    --map_name uv_seg  --cam_strategy smooth

python evaluate.py --gpu_ids 1 \
    --model imitator \
    --gen_name impersonator_adain_warp \
    --name impersonator_03_adain \
    --repeat_num 9 \
    --output_dir /p300/poseGANs \
    --image_size 256  \
    --map_name uv_seg  --cam_strategy source


python evaluate.py --gpu_ids 4 \
    --model imitator \
    --gen_name impersonator_adain_warp \
    --name impersonator_03_adain \
    --repeat_num 9 \
    --output_dir /p300/poseGANs \
    --image_size 256  \
    --map_name uv_seg  --cam_strategy copy