# OSNet-IBN (width x 1.0) Lite



## Contents

This project is a reduced version of [OSNet](https://arxiv.org/pdf/1905.00953.pdf)(Omni-Scale Feature Learning for Person Re-Identification), a network designed by Kaiyang Zhou. It has been extracted from [Torxreid](https://github.com/KaiyangZhou/deep-person-reid), the main framework for training and testing reID CNNs. 

Many features from the original backend have been isolated. Torxvision dependency has been deleted to keep docker container as light as possible. The required methods have been extracted from the source code. 

The net is able to work both in GPU and CPU. 



## Requirements

The required packages have been version-pinned in the *requirements.txt*.

`torch==1.2`
`numpy==1.16.4`
`Pillow`
`pandas`



## Dockerfile

This version includes two Dockerfiles: 

- `dev-nogpu.dockerfile`
- `dev-gpu.dockerfile`: CUDA 10 // CUDNN 7

For testing the GPU container, the user could use the following command: 

`docker run --runtime=nvidia  -it --volume /home/ec2-user/reid:/opt/project reid-dev:gpu /bin/bash`

Or a docker-compose file could be used. 



## Weights

Default weights are provided by the author at [Torxreid model zoo](https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO). 



## How to run it

The script main.py has the parameters that the user can modify. Unfortunately, in the present version, parameters have to be hard coded in the script.

A future implementation will include a better way of changing the parameters. 

The net ingests cropped images, detections, bounding boxes with a requested format. The format is shown in the 1<sup>st</sup> point.



1.- Load a pandas dataframe:

    # Load pandas dataframe
    df = pd.read_csv("./dataset/input_dataset.csv")

One of the columns has to be a string of a dict with the following format

    img_dict = {
        "width": image_width,
        "height": image_height,
        "colors": colors,
        "image": zlibed.hex()
    }

Where the image "zlibed", in bytes format, has to be converted to hexadecimal format. 



2.- Uncompress the cropped image:

    # Uncompress cropped image
    df["uncompressed_feature_vector"] = df.apply(lambda x: uncompress_string_image(
        compresed_cropped_image=x["feature_vector"]),
        axis=1)



3.- Declare the encoder object. 

    # Declare an encoder object
    encoder = OsNetEncoder(
        input_width=704,
        input_height=480,
        weight_filepath="weights/model_weights.pth.tar-40",
        batch_size=32,
        num_classes=2022,
        patch_height=256,
        patch_width=128,
        norm_mean=[0.485, 0.456, 0.406],
        norm_std=[0.229, 0.224, 0.225],
        GPU=True)
  - `num_classes`: Number of ids originally trained (CUHK03: 767, Market1501: 751)



4.- Register the new column as a new dataframe feature:      

    # Add the new column
    df["feature_vector"] = encoder.get_features(list(df["uncompressed_feature_vector"]))
    # Clean the dataframe
    df.drop("uncompressed_feature_vector", axis=1, inplace=True)



5.- Save the data as a .csv: 

    # Write the dataframe to a .csv
    df.to_csv("./output_files/output_dataset.csv",
    index=False,
    quoting=csv.QUOTE_NONNUMERIC
    )



## Performance

| CPU (2,2 GHz Intel Core i7):  | GPU (Tesla K80) |
| ------------- | ------------- |
| 2 Hz  | 35 Hz  |


## Citation


    @article{torchreid,
      title={Torchreid: A Library for Deep Learning Person Re-Identification in Pytorch},
      author={Zhou, Kaiyang and Xiang, Tao},
      journal={arXiv preprint arXiv:1910.10093},
      year={2019}
    }
    
    @inproceedings{zhou2019osnet,
      title={Omni-Scale Feature Learning for Person Re-Identification},
      author={Zhou, Kaiyang and Yang, Yongxin and Cavallaro, Andrea and Xiang, Tao},
      booktitle={ICCV},
      year={2019}
    }
    
    @article{zhou2019learning,
      title={Learning Generalisable Omni-Scale Representations for Person Re-Identification},
      author={Zhou, Kaiyang and Yang, Yongxin and Cavallaro, Andrea and Xiang, Tao},
      journal={arXiv preprint arXiv:1910.06827},
      year={2019}
    }
