python debug.py --gpu_ids 2 \
    --model liquid \
    --dataset_mode  fast_mi \
    --name debug \
    --image_size 512 \
    --map_name uv_seg \
    --batch_size 2  --use_vgg  --use_face --wrap  --do_saturate_mask

python debug.py --gpu_ids 3 \
    --model impersonator \
    --dataset_mode  fast_mi \
    --name debug \
    --image_size 256 \
    --map_name uv_seg \
    --batch_size 2  --use_vgg  --use_face


python debug.py --gpu_ids 0 \
    --model impersonator_02 \
    --dataset_mode  fast_mi \
    --name debug \
    --image_size 256 \
    --map_name uv_seg \
    --batch_size 2  --use_vgg  --use_face  --do_saturate_mask