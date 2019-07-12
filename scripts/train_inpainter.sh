###### impersonator trainer,  ti
python train.py --gpu_ids 0,1,2,3,4,5,6,7 \
    --data_dir  /public/liuwen/p300/human_pose/processed \
    --place_dir  /public/liuwen/p300/places365_standard \
    --checkpoints_dir  /public/liuwen/p300/models \
    --model background_inpaintor \
    --name background_inpaintor \
    --dataset_mode  smpl_place \
    --batch_size 80  \
    --train_G_every_n_iterations 1  \
    --nepochs_no_decay 40  --nepochs_decay 10 \
    --lr_G  0.0001 --lr_D 0.0004  \
    --final_lr 0.00001
