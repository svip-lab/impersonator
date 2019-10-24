# Training iPER dataset

### 1. Donwload the dataset
1. Download all files listed in [OneDrive](https://onedrive.live.com/?authkey=%21AJL_NAQMkdXGPlA&id=3705E349C336415F%2188052&cid=3705E349C336415F),
including `iPER_256_video_release.zip`, `smpls.zip`, `train.txt`, and `val.txt`.
Saving them into one folder, such as `root_dir=/p300/data`.

2. Extract the `smpls.zip` and `iPER_256_video_release.zip`.

3. Convert the video file into frames by the [script](../tools/unzip_iPER.py).
    Replace the path firstly,
    ```bash
   dataset_video_root_path = '/p300/data/iPER_256_video_release'
   save_images_root_path = '/p300/data/images'
    ```
    then, run the script
    ```bash
    python tools/unzip_iPER.zip
    ```
4. Format the folder tree as follows:
    ```shell
    |-- images
    |   |-- 001
    |   |-- 002
    |   .......
    |   |-- 029
    |   `-- 030
    |-- smpls
    |   |-- 001
    |   |-- 002
    |   .......
    |   |-- 029
    |   `-- 030
    |-- train.txt
    |-- val.txt
    
    `train.txt`: the splits of the training set.
    `val.txt`: the splits of the validation set.
    `images`: contains the images (frames) of each video.
    `smpls`: contains the smpls of each video.
    ```
    
### 2. Run the training script
1. Replace the `gpu_ids`, `data_dir` and `checkpoints_dir` in [training script](../scripts/train_iPER.sh).
    ```bash
    
    #gpu_ids=0,1     # if using multi-gpus
    gpu_ids=1
    
    # dataset configs
    data_dir=xxxxxxxx # the folder path that saves the iPER dataset (formated as above).
    dataset_model=iPER  # use iPER dataset
    
    # saving configs
    checkpoints_dir=xxxxxxxxx   #  directory to save models, to be replaced!!!!!
    name=exp_iPER   # the directory is ${checkpoints_dir}/name, which is used to save the checkpoints.
    
    ```

2. Run the [training script](../scripts/train_iPER.sh).
    ```bash
    chmod a+x scripts/train_iPER.sh
    ./scripts/train_iPER.sh
    ```

3. Tensorboard visualization
    ```shell
    tensorboard --logdir=${checkpoints_dir}/exp_iPER  --port=10086
    ```

# Training iPER + Place2 dataset
While for the background, the background network $G_{BG}$ is trained in a
self-supervised way, which seems to overfit the background
from the training set. One way to improve the ability of background generalization is to use additional
images, such as [Place2](http://places2.csail.mit.edu/download.html) dataset, as the auxiliary loss
$L_{aux}$ in the training phase. Specifically, in each training iteration,
we sample mini-batch images from Place2 dataset,
denoted as $L_{aux}$, add human body silhouettes to them, and
denote the mask images as $\hat{I}_{aux}$. We use the paired ($\hat{I}_{aux}$,
$I_{aux}$) images with a perceptual loss to update parameters in
the $G_{BG}$ network. The $L_{aux}$ loss indeed improves the generalization
of background inpainting. It is worth noting that for a fair comparison, we do not use
this trick in experiments when comparing our method with
other baselines.

1. Download iPER and format the folder like above.

2. Download [Place2](http://data.csail.mit.edu/places/places365/places365standard_easyformat.tar) and extract the data.

3. Replace the `gpu_ids`, `data_dir`, `place_dir`, and `checkpoints_dir` in [training script](../scripts/train_iPER.sh).
    ```bash
    
    #gpu_ids=0,1     # if using multi-gpus
    gpu_ids=1
    
    # dataset configs
    data_dir=/p300/iccv/iPER_examples  # need to be replaced!!!!!
    place_dir=/p300/iccv/places365_standard  # need to be replaced!!!!!
        
    # saving configs
    checkpoints_dir=xxxxxxxxx   #  directory to save models, to be replaced!!!!!
    name=exp_iPER_place   # the directory is ${checkpoints_dir}/name, which is used to save the checkpoints.
    
    ```

4. Run the [training script](../scripts/train_iPER.sh).
    ```bash
    chmod a+x scripts/train_iPER_Place2.sh
    ./scripts/train_iPER_Place2.sh
    ```

5. Tensorboard visualization
    ```shell
    tensorboard --logdir=${checkpoints_dir}/exp_iPER_place  --port=10086
    ```

# Training on other datasets
TODO