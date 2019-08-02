# Impersontor
PyTorch implementation of our ICCV 2019 paper:

Liquid Warping GAN: A Unified Framework for Human Motion Imitation, Appearance Transfer and Novel View Synthesis

## Getting Started
### Requirements
```
Python 3.6+ and PyTorch 1.0+.
```

**Note**: In some newer PyTorch versions you might see some compilation errors involving AT_ASSERT. In these cases you can use the version of the code that is in the branch *at_assert_fix*. These changes will be merged into master in the near future.

### Installation
```shell
cd thirdparty/neural_renderer
python setup.py install
```

### Run Demos and Examples
If the IP is http://10.10.10.100, and the port is 31102
```shell
python -m visdom.server --port 31102
```

#### Example 1: Motion Imitation
```
python demo_imitator.py --gpu_ids 0 \
    --model imitator_v2 \
    --gen_name impersonator \
    --name impersonator_mi \
    --checkpoints_dir /public/liuwen/p300/models \
    --output_dir /root/poseGANs/meta_train/impersonator_mi  \
    --image_size 256  --map_name uv_seg   \
    --ip http://10.10.10.100 --port 31102   \
    --bg_model /root/poseGANs/meta_train/background_inpaintor/net_epoch_50_id_G.pth \
    --src_path /root/poseGANs/meta_train/samples/all_img/men1_256.jpg    \
    --tgt_path /root/poseGANs/meta_train/samples/ref_imgs/2  \
    --pri_path /root/poseGANs/meta_train/samples/ref_imgs/1  \
    --has_detector  --bg_ks 7 --ft_ks 3  --front_warp --post_tune --visual
```
![motion imitation](asserts/motion_transfer.pdf)

#### Example 2: Appearance Transfer

```
python demo_swap.py --gpu_ids 9 \
    --model swapper_v2 \
    --gen_name impersonator \
    --name impersonator_mi \
    --checkpoints_dir /public/liuwen/p300/models \
    --output_dir '' \
    --image_size 256  --map_name uv_seg  --swap_part body  \
    --ip http://10.10.10.100 --port 31102   \
    --has_detector  --bg_ks 7 --ft_ks 3 --front_warp  --post_tune \
    --bg_model /root/poseGANs/meta_train/background_inpaintor/net_epoch_50_id_G.pth \
    --src_path  /root/poseGANs/meta_train/samples/all_img/men1_256.jpg \
    --tgt_path  /root/poseGANs/meta_train/samples/all_img/8_256.jpg  --visual
```
![appearance transfer](asserts/swap.pdf)

#### Example 3: Novel View Synthesis
```
python demo_view.py --gpu_ids 9 \
    --model viewer_v2 \
    --gen_name impersonator \
    --name impersonator_mi \
    --checkpoints_dir /public/liuwen/p300/models \
    --output_dir /root/poseGANs/meta_train/impersonator_mi \
    --image_size 256  --map_name uv_seg  --swap_part body  \
    --ip http://10.10.10.100 --port 31102   \
    --has_detector  --bg_ks 7 --ft_ks 3   --front_warp --post_tune \
    --bg_model /root/poseGANs/meta_train/background_inpaintor/net_epoch_50_id_G.pth \
    --src_path  /root/poseGANs/meta_train/samples/all_img/men1_256.jpg \
    --tgt_path  /root/poseGANs/meta_train/samples/all_img/8_256.jpg  --visual
```
![novel view](asserts/novel_view.pdf)

## Citation
```
@InProceedings{lwb2019,
    title={Liquid Warping GAN: A Unified Framework for Human Motion Imitation, Appearance Transfer and Novel View Synthesis},
    author={Wen Liu and Zhixin Piao, Min Jie, Wenhan Luo, Lin Ma and and Shenghua Gao},
    booktitle={The IEEE International Conference on Computer Vision (ICCV)},
    year={2019}
}
```
