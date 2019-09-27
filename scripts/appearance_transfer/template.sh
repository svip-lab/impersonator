#! /bin/bash

# internet <-> fashion
python run_swap.py --gpu_ids 0 --model imitator --output_dir ./outputs/results/  \
    --src_path      ./assets/src_imgs/imper_A_Pose/024_8_2_0000.jpg    \
    --tgt_path      ./assets/src_imgs/fashion_man/Sweatshirts_Hoodies-id_0000680701_4_full.jpg    \
    --bg_ks 13  --ft_ks 3 \
    --has_detector  --post_tune  --front_warp --swap_part body  \
    --save_res --ip http://10.10.10.100 --port 31102

