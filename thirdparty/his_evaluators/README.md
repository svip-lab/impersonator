### Human Image Synthesize Evaluators (his_evaluators)
This is a package of evaluators for human image synthesize, including human motion imitation, appearance transfer, and novel view synthesize.

### Update News
- [x] 04/07/2020, Human Imitation Imitation on iPER dataset, and see the evaluation 
protocol, [iPER_protocol.json](./data/iPER_protocol.json).

- [x] 04/28/2020, Add metric to evaluate the cropped person, and face.

### Installation
```shell
cd thirdparty/his_evaluators
pip install -e .
```

### Download pre-trained models
Manually download [evaluation resources](https://onedrive.live.com/?authkey=%21AHsa8jUFjDfXBzc&id=303FB25922AAD438%2172383&cid=303FB25922AAD438), and 
move the resources into `./data`.
### Usage

#### 1. Example
To use the evaluator, we must firstly implement all the interfaces 
of Class [MotionImitationModel](./his_evaluators/evaluators/motion_imitation.py), and 
```python
from his_evaluators import MotionImitationModel, IPERMotionImitationEvaluator

class LWGEvaluatorModel(MotionImitationModel):

    def __init__(self, opt, output_dir):
        super().__init__(output_dir)

        self.opt = opt
        
        # must declare the model in self.build_model()
        self.model = None

    def imitate(self, src_infos: Dict[str, Any], ref_infos: Dict[str, Any]) -> List[str]:
        """
            Running the motion imitation of the self.model, based on the source information with respect to the
            provided reference information. It returns the full paths of synthesized images.
        Args:
            src_infos (dict): the source information contains:
                --images (list of str): the list of full paths of source images (the length is 1)
                --smpls (np.ndarray): (length of images, 85)
                --kps (np.ndarray): (length of images, 19, 2)
            ref_infos (dict): the reference information contains:
                --images (list of str): the list of full paths of reference images.
                --smpls (np.ndarray): (length of images, 85)
                --kps (np.ndarray): (length of images, 19, 2)
                --self_imitation (bool): the flag indicates whether it is self-imitation or not.

        Returns:
            preds_files (list of str): full paths of synthesized images with respects to the images in ref_infos.
        """

        tgt_paths = ref_infos["images"]
        tgt_smpls = ref_infos["smpls"]
        self_imitation = ref_infos["self_imitation"]
        if self_imitation:
            cam_strategy = "copy"
            count = self.num_preds_si
            out_dir = self.si_out_dir
            self.num_preds_ci += len(tgt_paths)
        else:
            cam_strategy = "smooth"
            count = self.num_preds_ci
            out_dir = self.ci_out_dir
            self.num_preds_si += len(tgt_paths)
        outputs = self.model.inference(tgt_paths, tgt_smpls=tgt_smpls, cam_strategy=cam_strategy,
                                       visualizer=None, verbose=True)

        all_preds_files = []
        for i, preds in enumerate(outputs):
            filename = "{:0>8}.jpg".format(count)
            pred_file = os.path.join(out_dir, 'pred_' + filename)
            count += 1

            cv_utils.save_cv2_img(preds, pred_file, normalize=True)
            all_preds_files.append(pred_file)

        return all_preds_files

    def build_model(self):
        """
            You must define your model in this function, including define the graph and allocate GPU.
            This function will be called in @see `MotionImitationRunnerProcessor.run()`.
        Returns:
            None
        """
        # set imitator
        self.model = Imitator(self.opt)

    def personalization(self, src_infos):
        """
            some task/method specific data pre-processing or others.
        Args:
            src_infos (dict): the source information contains:
                --images (list of str): the list of full paths of source images (the length is 1)
                --smpls (np.ndarray): (length of images, 85)
                --kps (np.ndarray): (length of images, 19, 2)

        Returns:
            processed_src_infos (dict): the source information contains:
                --images (list of str): the list of full paths of source images (the length is 1)
                --smpls (np.ndarray): (length of images, 85)
                --kps (np.ndarray): (length of images, 19, 2)
                ...
        """

        # 1. load the pretrain model
        self.model._load_params(self.model.generator, self.opt.load_path)

        # 2. post personalization
        if self.opt.post_tune:
            self.opt.src_path = src_infos["images"][0]
            adaptive_personalize(self.opt, self.model, self.visualizer)

        processed_src_infos = src_infos
        return processed_src_infos

    def terminate(self):
        """
            Close the model session, like if the model is based on TensorFlow, it needs to call sess.close() to
            dealloc the resources.
        Returns:

        """
        pass


if __name__ == "__main__":
    opt = TestOptions().parse()

    model = LWGEvaluatorModel(opt, output_dir="/p300/iccv/baselines/WarpingStrategy/LWB-add/evaluations/iPER")
    iPER_MI_evaluator = IPERMotionImitationEvaluator(dataset="iPER", data_dir="/p300/iccv/iPER")
    
    # set dataset="iPER_ICCV" is the evaluation protocol of the previous ICCV version.
    # iPER_MI_evaluator = IPERMotionImitationEvaluator(dataset="iPER_ICCV", data_dir="/p300/iccv/iPER")

    iPER_MI_evaluator.evaluate(
        model=model,
        image_size=opt.image_size,
        pair_types=("ssim", "psnr", "lps"),
        unpair_types=("is", "fid", "PCB-CS-reid", "PCB-freid", "OS-CS-reid", "OS-freid")
    )

```

See the whole script in [evaluate.py](../../evaluate.py) for the reference.

#### 2. Metrics
##### 2.1 Motion Imitation
Here, we support self-imitation and cross-imitation metrics.

For self-imitation (pair_types), it provides:
* `SSIM`: we use the [skimage.metrics.structural_similarity](https://scikit-image.org/docs/dev/api/skimage.metrics.html#skimage.metrics.structural_similarity) to
calculate the similarity between the synthesized or generated image and ground truth image. Higher value is better.
* `PSNR`: [skimage.metrics.peak_signal_noise_ratio](https://scikit-image.org/docs/dev/api/skimage.metrics.html#skimage.metrics.peak_signal_noise_ratio) is applied to
calculate the peak signal noise ratio (PSNR). Higher value is better.
* `LPS`(or `LPIPS`): Using a learned perceptual similarity, [PerceptualSimilarity](https://github.com/richzhang/PerceptualSimilarity) to calculate the distance between the synthesized image and ground truth
image. Lower value is better.
* `OS-CS-reid`: This is the distance of the `cropped person region` between the 
synthesized image and the ground truth image. In particular, it firstly use [YoLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3)
detector to get the person bounding box of the synthesized and ground truth image. 
Then, we crop the person patches according to the bounding boxes. 
Finally, we use a pre-trained person-reid model, [OS-Net](https://github.com/KaiyangZhou/deep-person-reid),
to get the embedding features of the cropped person patches, normalize the features, and calculate the cosine similarity between the normalized features.
Higher value is better.

* `face-CS`: This is the distance of the `cropped face region` between the synthesized image and the ground truth image.
In particular, it firstly use [MTCNN](https://github.com/timesler/facenet-pytorch) 
face detector to get the face bounding boxes of the synthesized and ground truth images.
Then, we crop the face regions according to the bounding boxes. Finally, we use a 
pre-trained face recognition model, [InceptionResnetV1](https://github.com/timesler/facenet-pytorch),
to get the embedding features of the cropped face patches, normalize the features, and calculate the cosine similarity between the normalized features.
Higher value is better.

In general, `SSIM`, `PSNR`, and `LPS`(`LPIPS`) focus on the quality of the whole(global) synthesized images. `OS-CS-reid` 
focuses on the cropped front person region of the synthesized images, and `face-CS` focuses on the cropped face if the synthesized images.

For cross-imitation (unpair_types), 
1. for each video with person id `i`, we denote the input source image of person `i` as 
![](https://latex.codecogs.com/gif.latex?I^i_s)
or <a href="https://www.codecogs.com/eqnedit.php?latex=\{I^i_{s_1},&space;...,&space;I^i_{s_n}\}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\{I^i_{s_1},&space;...,&space;I^i_{s_n}\}" title="\{I^i_{s_1}, ..., I^i_{s_n}\}" /></a>;

2. we sample n consective frames from the source video with person `i`, and denote them as 
<a href="https://www.codecogs.com/eqnedit.php?latex=\{I^i_{1},&space;...,&space;I^i_{n}\}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\{I^i_{1},&space;...,&space;I^i_{n}\}" title="\{I^i_{1}, ..., I^i_{n}\}" /></a>;

3. we sample n consective frames from the reference video with person `j`, and denote them as 
<a href="https://www.codecogs.com/eqnedit.php?latex=\{I^j_1,&space;...,&space;I^j_n&space;\}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\{I^j_1,&space;...,&space;I^j_n&space;\}" title="\{I^j_1, ..., I^j_n \}" /></a>.

4. denoting the synthesized (imitated) images of the source input images of person `i` with respect to the reference images of person `j`
as <a href="https://www.codecogs.com/eqnedit.php?latex=\{I^{i\to&space;j}_1,&space;...,&space;I^{i\to&space;j}_n&space;\}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\{I^{i\to&space;j}_1,&space;...,&space;I^{i\to&space;j}_n&space;\}" title="\{I^{i\to j}_1, ..., I^{i\to j}_n \}" /></a>.

it provides the followings metrics:
* `is`: inception score(InceptionV3 backbone) of the synthesized images, <a href="https://www.codecogs.com/eqnedit.php?latex=\{I^{i\to&space;j}_1,&space;...,&space;I^{i\to&space;j}_n&space;\}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\{I^{i\to&space;j}_1,&space;...,&space;I^{i\to&space;j}_n&space;\}" title="\{I^{i\to j}_1, ..., I^{i\to j}_n \}" /></a>.

* `fid`: Frechet Inception Distance(InceptionV3 backbone) between the sampled source images (real) <a href="https://www.codecogs.com/eqnedit.php?latex=\{I^i_{1},&space;...,&space;I^i_{n}\}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\{I^i_{1},&space;...,&space;I^i_{n}\}" title="\{I^i_{1}, ..., I^i_{n}\}" /></a>;
and the synthesized images (fake)  <a href="https://www.codecogs.com/eqnedit.php?latex=\{I^{i\to&space;j}_1,&space;...,&space;I^{i\to&space;j}_n&space;\}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\{I^{i\to&space;j}_1,&space;...,&space;I^{i\to&space;j}_n&space;\}" title="\{I^{i\to j}_1, ..., I^{i\to j}_n \}" /></a>.
Lower value is better;

* `OS-CS-reid`: the cosine similarity of the `cropped person regions` betwwen the sampled source images (real) <a href="https://www.codecogs.com/eqnedit.php?latex=\{I^i_{1},&space;...,&space;I^i_{n}\}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\{I^i_{1},&space;...,&space;I^i_{n}\}" title="\{I^i_{1}, ..., I^i_{n}\}" /></a>;
and the synthesized images (fake) <a href="https://www.codecogs.com/eqnedit.php?latex=\{I^{i\to&space;j}_1,&space;...,&space;I^{i\to&space;j}_n&space;\}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\{I^{i\to&space;j}_1,&space;...,&space;I^{i\to&space;j}_n&space;\}" title="\{I^{i\to j}_1, ..., I^{i\to j}_n \}" /></a>,
and the details are shown in the above. Higher value is better;

* `PCB-CS-reid`: This is similar as the `OS-CS-reid`, and the difference is the backbone of person-reid model, here, we use
the [PCB-Net](https://github.com/layumi/Person_reID_baseline_pytorch). Higher value is better;

* `face-CS`: the cosine similarity of the `cropped face regions` betwwen the sampled source images (real) <a href="https://www.codecogs.com/eqnedit.php?latex=\{I^i_{1},&space;...,&space;I^i_{n}\}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\{I^i_{1},&space;...,&space;I^i_{n}\}" title="\{I^i_{1}, ..., I^i_{n}\}" /></a>;
and the synthesized images (fake) <a href="https://www.codecogs.com/eqnedit.php?latex=\{I^{i\to&space;j}_1,&space;...,&space;I^{i\to&space;j}_n&space;\}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\{I^{i\to&space;j}_1,&space;...,&space;I^{i\to&space;j}_n&space;\}" title="\{I^{i\to j}_1, ..., I^{i\to j}_n \}" /></a>,
and the details are shown in the above. Higher value is better;

* `OS-freid`: the Frechet Distance of [OS-Net](https://github.com/KaiyangZhou/deep-person-reid). In particular,
we firstly crop the person regions of the sampled source images (real) <a href="https://www.codecogs.com/eqnedit.php?latex=\{I^i_{1},&space;...,&space;I^i_{n}\}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\{I^i_{1},&space;...,&space;I^i_{n}\}" title="\{I^i_{1}, ..., I^i_{n}\}" /></a>;
and the synthesized images (fake)  <a href="https://www.codecogs.com/eqnedit.php?latex=\{I^{i\to&space;j}_1,&space;...,&space;I^{i\to&space;j}_n&space;\}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\{I^{i\to&space;j}_1,&space;...,&space;I^{i\to&space;j}_n&space;\}" title="\{I^{i\to j}_1, ..., I^{i\to j}_n \}" /></a> based on
the [YoLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3). Lower value is better;

* `PCB-freid`: the Frechet Distance of [PCB-Net](https://github.com/layumi/Person_reID_baseline_pytorch). In particular,
we firstly crop the person regions of the sampled source images (real) <a href="https://www.codecogs.com/eqnedit.php?latex=\{I^i_{1},&space;...,&space;I^i_{n}\}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\{I^i_{1},&space;...,&space;I^i_{n}\}" title="\{I^i_{1}, ..., I^i_{n}\}" /></a>;
and the synthesized images (fake)  <a href="https://www.codecogs.com/eqnedit.php?latex=\{I^{i\to&space;j}_1,&space;...,&space;I^{i\to&space;j}_n&space;\}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\{I^{i\to&space;j}_1,&space;...,&space;I^{i\to&space;j}_n&space;\}" title="\{I^{i\to j}_1, ..., I^{i\to j}_n \}" /></a> based on
the [YoLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3). It is very slow, O(c^3). n is the dimension of the feature of PCB-Net,
and c = 12,288‬. Lower value is better;

* `face-FD`: the Frechet Distance of cropped face between the sampled source images (real) <a href="https://www.codecogs.com/eqnedit.php?latex=\{I^i_{1},&space;...,&space;I^i_{n}\}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\{I^i_{1},&space;...,&space;I^i_{n}\}" title="\{I^i_{1}, ..., I^i_{n}\}" /></a>;
and the synthesized images (fake)  <a href="https://www.codecogs.com/eqnedit.php?latex=\{I^{i\to&space;j}_1,&space;...,&space;I^{i\to&space;j}_n&space;\}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\{I^{i\to&space;j}_1,&space;...,&space;I^{i\to&space;j}_n&space;\}" title="\{I^{i\to j}_1, ..., I^{i\to j}_n \}" /></a>.
Specifically, similar to `face-CS`, we firstly use [MTCNN](https://github.com/timesler/facenet-pytorch) to detect the face bounding boxes,
and then use [InceptionResnetV1](https://github.com/timesler/facenet-pytorch) to extract the feature embeddings, and calculate
the Frechet Distance based on the face embedding features. Lower value is better;

In general:
1. `is` seems not be a good metric for human image synthesize, because, `is` is based on the a pre-trained 
InceptionV3 network on ImageNet dataset, and `is` mainly focus on the diversity of generative models.
However, in human image synthesize, the outputs are always the `human` labels, it will result in very low performance;

2. `fid` focuses on the whole (global) synthesized images;

3. `OS-CS-reid`, `PCB-CS-reid`, `OS-freid`, and `PCB-freid` focus on the cropped front person region of the synthesized images;

4. `face-CS` and `face-FD` focus on the cropped face if the synthesized images.

#### Notice
Here, we fix some bugs on the previous implementation of `IS` and `PCB-freid` in the ICCV paper. This evaluation package is the 
most recent version, and the results are different from the results reported in the ICCV paper.
Our future extended journal paper will be based on this implementation. We recommend this implementation as the iPER evaluation.


#### Reference
* K. Zhang, Z. Zhang, Z. Li and Y. Qiao. Joint Face Detection and Alignment Using Multitask Cascaded Convolutional Networks, IEEE Signal Processing Letters, 2016.

* Zhou, Kaiyang and Yang, Yongxin and Cavallaro, Andrea and Xiang, Tao. Omni-Scale Feature Learning for Person Re-Identification, ICCV 2019.

* Yifan Sun, Liang Zheng, Yi Yang, Qi Tian, Shengjin Wang. Beyond Part Models: Person Retrieval with Refined Part Pooling (and a Strong Convolutional Baseline), ECCV 2018.

* Redmon, Joseph and Farhadi, Ali. YOLOv3: An Incremental Improvement, arxiv 2018.

* Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, Oliver Wang. The Unreasonable Effectiveness of Deep Features as a Perceptual Metric, CVPR 2018.

* [PCB-Net](https://github.com/layumi/Person_reID_baseline_pytorch);
* [OS-Net](https://github.com/KaiyangZhou/deep-person-reid);
* [YoLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3);
* [facenet-pytorch](https://github.com/timesler/facenet-pytorch).