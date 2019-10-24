# Impersonator
PyTorch implementation of our ICCV 2019 paper:

Liquid Warping GAN: A Unified Framework for Human Motion Imitation, Appearance Transfer and Novel View Synthesis

**Please clone the newest codes.**

[[paper]](https://arxiv.org/pdf/1909.12224.pdf) [[website]](https://svip-lab.github.io/project/impersonator)

<p float="center">
	<img src='assets/visuals/motion/Sweaters-id_0000088807_4_full.jpg' width="135"/>
  	<img src='assets/visuals/motion/mixamo_0007_Sweaters-id_0000088807_4_full.gif' width="135"/>
  	<img src='assets/visuals/appearance/Sweaters-id_0000337302_4_full.jpg' width="135"/>
	<img src='assets/visuals/appearance/Sweaters-id_0000337302_4_full.gif' width="135"/>
	<img src='assets/visuals/novel/Jackets_Vests-id_0000071603_4_full.jpg' width="135"/>
    <img src='assets/visuals/novel/Jackets_Vests-id_0000071603_4_full.gif' width="135"/>
    <img src='assets/visuals/motion/009_5_1_000.jpg' width="135"/>    
  	<img src='assets/visuals/motion/mixamo_0031_000.gif' width="135"/>
  	<img src='assets/visuals/appearance/001_19_1_000.jpg' width="135"/>
	<img src='assets/visuals/appearance/001_19_1_000.gif' width="135"/>
	<img src='assets/visuals/novel/novel_3.jpg' width="135"/>
    <img src='assets/visuals/novel/novel_3.gif' width="135"/>
</p>

## Update News
- [x] 10/05/2019, optimize the minimal requirements of GPU memory (at least `3.8GB` available).

- [x] 10/24/2019, Imper-1.2.2, add the training document [train.md](./doc/train.md).

## Getting Started
Python 3.6+, Pytorch 1.2, torchvision 0.4, cuda10.0, at least `3.8GB` GPU memory and other requirements.
All codes are tested on Linux Distributions (Ubutun 16.04 is recommended), and other platforms have not been tested yet.

### Requirements
``` bash
pip install -r requirements.txt
```

### Installation
```shell
cd thirdparty/neural_renderer
python setup.py install
```

### Download resources.
1. Download `pretrains.zip` from [OneDrive](https://1drv.ms/u/s!AjjUqiJZsj8whLNw4QyntCMsDKQjSg?e=L77Elv) or
[BaiduPan](https://pan.baidu.com/s/11S7Z6Jj3WAfVNxBWyBjW6w) and then move the pretrains.zip to 
the `assets` directory and unzip this file.
```
wget -O assets/pretrains.zip https://1drv.ws/u/s!AjjUqiJZsj8whLNw4QyntCMsDKQjSg?e=L77Elv
```

2. Download `checkpoints.zip` from [OneDrive](https://1drv.ms/u/s!AjjUqiJZsj8whLNyoEh67Uu0LlxquA?e=dkOnhQ) or
[BaiduPan](https://pan.baidu.com/s/1snolk6wphbuHtQ_DeSA06Q) and then 
unzip the `checkpoints.zip` and move them to `outputs` directory.
```
wget -O outputs/checkpoints.zip https://1drv.ws/u/s!AjjUqiJZsj8whLNyoEh67Uu0LlxquA?e=dkOnhQ
```

3. Download `samples.zip` from [OneDrive](https://1drv.ms/u/s!AjjUqiJZsj8whLNz4BqnSgqrVwAXoQ?e=bC86db) or
[BaiduPan](https://pan.baidu.com/s/1xAI96709Gvqahq9uYAEXYA), and then
unzip the `samples.zip` and move them to `assets` directory.
```
wget -O assets/samples.zip https://1drv.ws/u/s!AjjUqiJZsj8whLNz4BqnSgqrVwAXoQ?e=bC86db
```

### Running Demo
If you want to get the results of the demo shown on the webpage, you can run the following scripts.
The results are saved in `./outputs/results/demos`

1. Demo of Motion Imitation
    ```bash
    python demo_imitator.py --gpu_ids 1
    ```
    
2. Demo of Appearance Transfer
    ```bash
    python demo_swap.py --gpu_ids 1
    ```

3. Demo of Novel View Synthesis
    ```bash
    python demo_view.py --gpu_ids 1
    ```
    
If you get the errors like `RuntimeError: CUDA out of memory`, please add the flag `--batch_size 1`, the minimal 
GPU memory is 3.8 GB.


### Running custom examples (Details)
If you want to test other inputs (source image and reference images **from yourself**), here are some examples.
Please replace the `--ip YOUR_IP` and `--port YOUR_PORT` for 
[Visdom](https://github.com/facebookresearch/visdom) visualization. 

1. Motion Imitation
    * source image from iPER dataset
    ```bash
    python run_imitator.py --gpu_ids 0 --model imitator --output_dir ./outputs/results/  \
        --src_path      ./assets/src_imgs/imper_A_Pose/009_5_1_000.jpg    \
        --tgt_path      ./assets/samples/refs/iPER/024_8_2    \
        --bg_ks 13  --ft_ks 3 \
        --has_detector  --post_tune  \
        --save_res --ip YOUR_IP --port YOUR_PORT
    ```
        
    * source image from DeepFashion dataset
    ```bash
    python run_imitator.py --gpu_ids 0 --model imitator --output_dir ./outputs/results/  \
    --src_path      ./assets/src_imgs/fashion_woman/Sweaters-id_0000088807_4_full.jpg    \
    --tgt_path      ./assets/samples/refs/iPER/024_8_2    \
    --bg_ks 25  --ft_ks 3 \
    --has_detector  --post_tune  \
    --save_res --ip YOUR_IP --port YOUR_PORT
    ```
        
    * source image from Internet
    ```bash
    python run_imitator.py --gpu_ids 0 --model imitator --output_dir ./outputs/results/  \
        --src_path      ./assets/src_imgs/internet/men1_256.jpg    \
        --tgt_path      ./assets/samples/refs/iPER/024_8_2    \
        --bg_ks 7   --ft_ks 3 \
        --has_detector  --post_tune --front_warp \
        --save_res --ip YOUR_IP --port YOUR_PORT
    ```
2. Appearance Transfer

    An example that source image from iPER and reference image from DeepFashion dataset.

    ```bash
    python run_swap.py --gpu_ids 0 --model imitator --output_dir ./outputs/results/  \
        --src_path      ./assets/src_imgs/imper_A_Pose/024_8_2_0000.jpg    \
        --tgt_path      ./assets/src_imgs/fashion_man/Sweatshirts_Hoodies-id_0000680701_4_full.jpg    \
        --bg_ks 13  --ft_ks 3 \
        --has_detector  --post_tune  --front_warp --swap_part body  \
        --save_res --ip http://10.10.10.100 --port 31102
    ```
3. Novel View Synthesis
    ```bash
    python run_view.py --gpu_ids 0 --model viewer --output_dir ./outputs/results/  \
    --src_path      ./assets/src_imgs/internet/men1_256.jpg    \
    --bg_ks 13  --ft_ks 3 \
    --has_detector  --post_tune --front_warp --bg_replace \
    --save_res --ip http://10.10.10.100 --port 31102
    ```
    
If you get the errors like `RuntimeError: CUDA out of memory`, please add the flag `--batch_size 1`, the minimal 
GPU memory is 3.8 GB.

The details of each running scripts are shown in [runDetails.md](doc/runDetails.md).
### Training from Scratch

* The details of training iPER dataset from scratch are shown in [train.md](./doc/train.md).

## Announcement
In our paper, the results of LPIPS reported in Table 1, are calculated by **1 – distance score**; 
thereby, the larger is more similar between two images. The beginning intention of using **1 – distance score** is that it is more accurate to meet the definition of **Similarity** in LPIPS.

However, most other papers use the original definition that LPIPS = distance score; 
therefore, to eliminate the ambiguity and make it consistent with others, 
we update the results in Table 1 with the original definition in the [latest paper](https://arxiv.org/pdf/1909.12224.pdf).

## Citation
![thunmbnail](assets/thumbnail.jpg)
```
@InProceedings{lwb2019,
    title={Liquid Warping GAN: A Unified Framework for Human Motion Imitation, Appearance Transfer and Novel View Synthesis},
    author={Wen Liu and Zhixin Piao, Min Jie, Wenhan Luo, Lin Ma and Shenghua Gao},
    booktitle={The IEEE International Conference on Computer Vision (ICCV)},
    year={2019}
}
```
