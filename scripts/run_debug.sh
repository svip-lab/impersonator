python debug.py --gpu_ids 8 \
    --data_dir  /public/liuwen/p300/human_pose/processed \
    --place_dir  /public/liuwen/p300/places365_standard \
    --checkpoints_dir  /public/liuwen/p300/models \
    --model impersonator \
    --name impersonator \
    --dataset_mode  mi_v2 \
    --image_size 256 \
    --map_name uv_seg \
    --batch_size 4  --use_vgg  --use_face --only_visible \
    --lambda_face 5.0  --lambda_mask 0.5 \
    --train_G_every_n_iterations 5  \
    --nepochs_no_decay 10  --nepochs_decay 40 \
    --train_ids_file  MI_val_debug.txt  \
    --test_ids_file   MI_val_debug.txt


python debug.py --gpu_ids 8 \
    --data_dir  /public/liuwen/p300/human_pose/processed \
    --place_dir  /public/liuwen/p300/places365_standard \
    --checkpoints_dir  /public/liuwen/p300/models \
    --model impersonator \
    --name impersonator \
    --dataset_mode  mi_v2 \
    --image_size 256 \
    --map_name uv_seg \
    --batch_size 2  --use_vgg  --use_face --wrap_face \
    --lambda_face 5.0  --lambda_mask 0.5 \
    --train_G_every_n_iterations 5  \
    --nepochs_no_decay 10  --nepochs_decay 40 \
    --train_ids_file  MI_val.txt  \
    --test_ids_file   MI_val.txt