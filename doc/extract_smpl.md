# 1. Extract the frames of each video in `SOURCE_VIDEOS_DIR`  by ffmpeg as saved into `SOURCE_FRAMES_DIR`;

```shell
# the diretory contains the videos of your dataset
SOURCE_VIDEOS_DIR :
  -----video_1.mp4
  -----video_2.mp4
    ....
  -----video_n.mp4
```
```shell
# the diretory contains the extracted frames of each video of your dataset
SOURCE_FRAMES_DIR:
-----video_1:
---------------: frame0000000.png
---------------: frame0000001.png
.....
-----video_2:
.....
```

# 3.2 `!!!!!!!!!!!!!!!!!!!!! important !!!!!!!!!!!!!!!!!!!!!!!`, human center crop and pad each image/frame into a `square image`.
This step is very important, because the human body recovery model (HMR)  will achieve a better result of SMPL when the input is square image with human center cropped.
You can manually give a rough square bounding boxes of each video or automatically run the person detector to get the person bounding boxes of each frame, and crop the original frames based on the bounding boxes with padding,  and then you will get the human center cropped square image. Saving the cropped images into `OUTPUT_DIR`.
```shell
# the diretory contains the estimated SMPL data for training, and the format is same as iPER dataset.
OUTPUT_DIR:
-----images:
---------video_1:
---------------: frame0000000.png
---------------: frame0000001.png
....
```
# 3. Use the hmr to estimate the SMPL data of each frame, and format the data like iPER dataset.
```python
import os
import torch
import numpy as np
import cv2
import glob
import pickle

from networks.hmr import HumanModelRecovery


OUTPUT_DIR = "the diretory contains the estimated SMPL data for training, and the format is same as iPER dataset"



device = torch.device("cuda:0")

# use hmr to estimate the smpl of each frame
hmr = HumanModelRecovery("./assets/pretrains/smpl_model.pkl")
saved_data = torch.load("./assets/pretrains/hmr_tf2pt.pth")
hmr.load_state_dict(saved_data).to(device)
hmr.eval()


def split(theta):
    """
    Args:

        theta: (N, 3 + 72 + 10)

    Returns:

    """

    cam = theta[:, 0:3]
    pose = theta[:, 3:-10]
    shape = theta[:, -10:]

    return cam, pose, shape


# for-loop on each video
for video in os.listdir(OUTPUT_DIR):
    vid_dir = os.path.join(OUTPUT_DIR, video)

    # smpl data of each video
    smpl_info = {
        "cams": [],
        "pose": [],
        "shape": []
    }

    images_paths = glob.glob(os.path.join(vid_dir, "images", "*"))
    images_paths.sort()

    for img_path in images_paths:
        # !! assume the image has been human-center cropped and padded into a square image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224)) # hmr receive 224 x 224 inputs
        img = np.transpose(2, 0, 1)
        img = img.astype(np.float32)
        img = img / 255 * 2 - 1 # [-1, 1]

        img = torch.tensor(img).float()[None]
        smpls = hmr.forward(img) # [1, 85]
        cam, pose, shape = split(smpls.cpu().numpy())

        smpl_info["cams"].append(cam)
        smpl_info["pose"].append(pose)
        smpl_info["shape"].append(shape)


    smpl_info["cams"] = np.concatenate(smpl_info["cams"], axis=0)
    smpl_info["pose"] = np.concatenate(smpl_info["pose"], axis=0)
    smpl_info["shape"] = np.concatenate(smpl_info["shape"], axis=0)

    # write it into disk
    with open(os.path.join(vid_dir, "smpls", "pose_shape.pkl"), "rb") as writer:
        pickle.dump(smpl_info, writer, protocol=2)
```

The diretory contains the estimated SMPL data for training, and the format is similar to iPER dataset, and you can modify the 

`train.txt` and `val.txt`  to train your dataset.

```shell

OUTPUT_DIR:
-----images:
---------video_1:
---------------: frame0000000.png
---------------: frame0000001.png
....
-----smpls:
---------video_1:
---------------: pose_shape.pkl
....
```