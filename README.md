# Impersontor
PyTorch implementation of our ICCV 2019 paper:

Liquid Warping GAN: A Unified Framework for Human Motion Imitation, Appearance Transfer and Novel View Synthesis

## Getting Started
### Requirements
``` bash
Python 3.6+ and PyTorch 1.0+.

pip install -r requirements.txt
```

### Installation
```shell
cd thirdparty/neural_renderer
python setup.py install
```

### Download resources.
1. Download `pretrains.zip` from [OneDrive]("https://1drv.ms/u/s!AjjUqiJZsj8whLNw4QyntCMsDKQjSg?e=L77Elv") or
[BaiduPan]("https://pan.baidu.com/s/11S7Z6Jj3WAfVNxBWyBjW6w") and then move the pretrains.zip to 
the `assets` directory and unzip this file.

2. Download `checkpoints.zip` from [OneDrive]("https://1drv.ms/u/s!AjjUqiJZsj8whLNyoEh67Uu0LlxquA?e=dkOnhQ") and then 
unzip the `checkpoints.zip` and move them to `outputs` directory.

3. Download `samples.zip` from [OneDrive]("https://1drv.ms/u/s!AjjUqiJZsj8whLNxCKkPaJnqxbbodQ?e=40uty2"), and then
unzip the `samples.zip` and move them to `assets` directory.

### Run Demos and Examples
The details are shown in [runDemo.md](./doc/runDemo.md).

### Training from Scratch
The details are shown in [train.md](./doc/train.md).

## Citation
![thunmbnail](assets/thumbnail.jpg)
```
@InProceedings{lwb2019,
    title={Liquid Warping GAN: A Unified Framework for Human Motion Imitation, Appearance Transfer and Novel View Synthesis},
    author={Wen Liu and Zhixin Piao, Min Jie, Wenhan Luo, Lin Ma and and Shenghua Gao},
    booktitle={The IEEE International Conference on Computer Vision (ICCV)},
    year={2019}
}
```
