### Human Image Synthesize Evaluators (his_evaluators)
This is a package of evaluators for human image synthesize, including human motion imitation, appearance transfer, and novel view synthesize.

### Update News
- [x] 04/07/2020, Human Imitation Imitation on iPER dataset, and see the evaluation 
protocol, [iPER_protocol.json](./data/iPER_protocol.json).

### Installation
```shell
cd thirdparty/his_evaluators
pip install -e .
```

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
    iPER_MI_evaluator = IPERMotionImitationEvaluator(data_dir="/p300/iccv/iPER")

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

For self-imitation (pair_types), it provides,
* `SSIM`:
* `PSNR`: 
* `LPS`(or `LPIPS`):

For cross-imitation (unpair_types), it provides,
* `is`: inception score, (InceptionV3 backbone);
* `fid`: Frechet Inception Distance (InceptionV3 backbone);
* `PCB-CS-reid`: the cosine similarity of a pre-trained person re-identification model, 
PCB [PCB-Net backbone](http://openaccess.thecvf.com/content_ECCV_2018/papers/Yifan_Sun_Beyond_Part_Models_ECCV_2018_paper.pdf).
* `PCB-freid`: the Frechet Distance of PCB-Net, and it is very slow, O(n^3). n is the dimension of the feature of PCB-Net,
and n = 12,288‬.
* `OS-CS-reid`: the cosine similarity of a pre-trained person re-identification model, 
[OS-Net](https://github.com/KaiyangZhou/deep-person-reid/blob/master/torchreid/models/osnet.py).
* `OS-freid`: the Frechet Distance of OS-Net.

#### Notice
Here, we fix some bugs on the previous implementation of `IS` and `PCB-freid` in the ICCV paper. This evaluation package is the 
most recent version, and the results are different from the results reported in the ICCV paper.
Our future extended journal paper will be based on this implementation. We recommend this implementation as the iPER evaluation.