#! /bin/bash

# iPER
python run_imitator.py --gpu_ids 3 --model swapper --output_dir ./outputs/results/  \
    --src_path      ./assets/src_imgs/imper_A_Pose/009_5_1_000.jpg    \
    --tgt_path      ./assets/samples/refs/iPER/024_8_2    \
    --bg_ks 13  --ft_ks 3 \
    --has_detector  --post_tune  \
    --save_res --ip http://10.10.10.100 --port 31102

# DeepFashion
python run_imitator.py --gpu_ids 0 --model swapper --output_dir ./outputs/results/  \
    --src_path      ./assets/src_imgs/fashion_woman/Sweaters-id_0000088807_4_full.jpg    \
    --tgt_path      ./assets/samples/refs/iPER/024_8_2    \
    --bg_ks 25  --ft_ks 3 \
    --has_detector  --post_tune  \
    --save_res --ip http://10.10.10.100 --port 31102

# Internet
python run_imitator.py --gpu_ids 0 --model swapper --output_dir ./outputs/results/  \
    --src_path      ./assets/src_imgs/internet/men1_256.jpg    \
    --tgt_path      ./assets/samples/refs/iPER/024_8_2    \
    --bg_ks 7  --ft_ks 3 \
    --has_detector  --post_tune --front_warp \
    --save_res --ip http://10.10.10.100 --port 31102
