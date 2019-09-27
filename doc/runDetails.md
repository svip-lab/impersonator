# Motion Imitation Examples
### Running Examples

1. Source image from Internet
    ```bash
    python run_imitator.py --gpu_ids 0 --model imitator --output_dir ./outputs/results/  \
        --src_path      ./assets/samples/src_imgs/men1_256.jpg    \
        --tgt_path      ./assets/samples/ref_imgs/024_8_2    \
        --has_detector  --post_tune   --front_warp  --save_res
    ```
    
2. Details of some important flags
    The basic running examples is [here](../scripts/motion_imitation/runner.sh), as shown in followings:
    ```bash
    #! /bin/bash
    
    # choose other inputs src img and reference images
    src_path="./assets/samples/src_imgs/men1_256.jpg"
    tgt_path="./assets/samples/ref_imgs/024_8_2"
    
    ##
    gpu=3
    gen_name="impersonator"
    name="imper_results"
    checkpoints_dir="./outputs/checkpoints/"
    output_dir="./outputs/results/"
    
    ## if use ImPer dataset trained model
    #load_path="./outputs/checkpoints/lwb_imper/net_epoch_30_id_G.pth"
    
    ## if use ImPer and Place datasets trained model
    #load_path="./outputs/checkpoints/lwb_imper_place/net_epoch_30_id_G.pth"
    
    ## if use ImPer, DeepFashion, and Place datasets trained model
    load_path="./outputs/checkpoints/lwb_imper_fashion_place/net_epoch_30_id_G.pth"
    
    ## if use DeepFillv2 trained background inpainting network,
    bg_model="./outputs/checkpoints/deepfillv2/net_epoch_50_id_G.pth"
    ## otherwise, it will use the BGNet in the original LiquidWarping GAN
    #bg_model="ORIGINAL"
    
    python demo_imitator.py --gpu_ids ${gpu} \
        --model imitator \
        --gen_name impersonator \
        --image_size 256 \
        --name ${name}  \
        --checkpoints_dir ${checkpoints_dir} \
        --bg_model ${bg_model}      \
        --load_path ${load_path}    \
        --output_dir ${output_dir}  \
        --src_path   ${src_path}    \
        --tgt_path   ${tgt_path}    \
        --bg_ks 7 --ft_ks 3         \
        --has_detector  --post_tune  --front_warp --save_res
    ```

* `--src_path` and `--tgt_path`: these are the path of source and reference images, respectively. The `tgt_path`
can be both a specific image or a directory contains a list of images.

* `output_dir`: output directory to save the results,
    
    1. the results of motion imitation will be saved in `${output_dir}/imitators`;
    2. the results of appearance transfer will be saved in `${output_dir}/swappers`;
    3. the results of novel view synthesis will be save in `${output_dir}/viewers`.

* `--save_res`: control whether to save the synthesized images or not, if true, then the results are saved 
in `${output_dir}/preds` folder.

* `--ip`, `--port` and `--name`: these two flags are the `ip` and `port` parameters used by `Visdom`
for online visualization. For example, you can add the flags at the end of the basic running
scripts, as the followings:
    ```bash
        ...
        --has_detector  --post_tune --save_res \
        --ip http://10.10.10.100 --port 31102
    ```
    Then you can open your local browser into `http://10.10.10.100:31102`, and the environment of `Visdom` is `name`.

* `--front_warp`: it is used to directly copy the front face (head). ***!!!Be careful*** to  this flag, 
because the `HMR` can not align well at head part of human body, it will result in the artifact that 
the synthesized image likes a MASK MAN.

* `--post_tune`: this flag is to control whether to do instance-adaptation or not. Since when given an arbitrary source 
image (out domain of the training set) from the Internet, the network seems to synthesize the style of images prone to the 
training set. The details of `post_tune`are shown in [here](./postTune.md).

* `--load_path`: load which model for inference.
    ```bash
    # if use ImPer dataset trained model
    --load_path ./outputs/checkpoints/lwb_imper/net_epoch_30_id_G.pth
    
    # if use ImPer and Place datasets trained model
    --load_path ./outputs/checkpoints/lwb_imper_place/net_epoch_30_id_G.pth
    
    # if use ImPer, DeepFashion, and Place datasets trained model (default)
    --load_path ./outputs/checkpoints/lwb_imper_fashion_place/net_epoch_30_id_G.pth 
    ```

* `--bg_model`: Whether to use DeepFillv2 trained background inpainting network (default is used) or not, otherwise it 
will use the BGNet in the original LiquidWarping GAN.

    If use DeepFillv2, modify the scripts as followings (default):
    ```bash
    ## if use DeepFillv2 trained background inpainting network,
    bg_model="./outputs/checkpoints/deepfillv2/net_epoch_50_id_G.pth"
    ## otherwise, it will use the BGNet in the original LiquidWarping GAN
    #bg_model="ORIGINAL"
    ```
    Else:
    ```bash
    ## if use DeepFillv2 trained background inpainting network,
    #bg_model="./outputs/checkpoints/deepfillv2/net_epoch_50_id_G.pth"
    ## otherwise, it will use the BGNet in the original LiquidWarping GAN
    bg_model="ORIGINAL"
    ```

* `--has_detector`: this flag is to control whether to use `Mask-rcnn` to estimate the body segmentation or not.
If it is true, it will use `Mask-rcnn` to detect the body segmentation, otherwise it will the estimated rendered 3D body 
silhouette by the `Neural Mesh Renderer` and the estimated SMPL parameters by `HMR`. In summary, use the `Mask-rcnn` 
will result in a more accurate body segmentation.

* `--bg_ks` and `-ft_ks`: the size of dilation kernel of background and front mask respectively. Since the estimated 
body segmentation might be smaller than the actual segmentation. Using the dilation operation to enlarge the mask.

# Run Appearance Transfer
[Here](../scripts/appearance_transfer/runner.sh) is the basic running scripts. 
The meaning of each flag is same to motion imitation.

* `--save_res`: If true, the results are saved in `${output_dir}/swappers` folder.

# Run Novel View Synthesis
[Here](../scripts/novel_view_synthesis/runner.sh) is the basic running scripts. 
The meaning of each flag is same to motion imitation.

* `--save_res`: If true, the results are saved in `${output_dir}/viewers` folder.

* `--bg_replace`: If true, it will use the original background, otherwise, it will use template background (gray).



