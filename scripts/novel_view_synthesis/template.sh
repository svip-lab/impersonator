#! /bin/bash

python run_view.py --gpu_ids 3 --model viewer --output_dir ./outputs/results/  \
    --src_path      ./assets/src_imgs/internet/men1_256.jpg    \
    --bg_ks 13  --ft_ks 3 \
    --has_detector  --post_tune --front_warp --bg_replace \
    --save_res --ip http://10.10.10.100 --port 31102

