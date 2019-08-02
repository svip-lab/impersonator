python debug.py --gpu_ids 9 \
    --model impersonator_02 \
    --dataset_mode  fast_mi \
    --name debug \
    --image_size 256 \
    --map_name uv_seg \
    --batch_size 2  --use_vgg  --use_face \
    --data_dir  /public/liuwen/p300/ImPer

python debug.py --gpu_ids 9 \
    --model impersonator_trainer \
    --dataset_mode  fast_mi \
    --name debug \
    --image_size 256 \
    --map_name uv_seg \
    --batch_size 2  --use_vgg  --use_face --face_model  \
    --data_dir  /public/liuwen/p300/ImPer