import torch
import numpy as np
import unittest


from his_evaluators.metrics import register_metrics


DEVICE = torch.device("cuda:0")


class MetricTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.paired_metric_dict = register_metrics(types=("ssim", "psnr", "lps"), device=DEVICE)
        cls.unpaired_metric_dict = register_metrics(
            types=("is", "fid", "PCB-CS-reid", "PCB-freid", "OS-CS-reid", "OS-freid"),
            device=DEVICE
        )

    def test_01_paired_metrics(self):
        bs = 5
        image_size = 512
        preds_imgs = np.random.rand(bs, 3, image_size, image_size)
        preds_imgs *= 255
        preds_imgs = preds_imgs.astype(np.uint8)
        ref_imgs = np.copy(preds_imgs)

        ssim_score = self.paired_metric_dict["ssim"].calculate_score(preds_imgs, ref_imgs)
        psnr_score = self.paired_metric_dict["psnr"].calculate_score(preds_imgs, ref_imgs)
        lps_score = self.paired_metric_dict["lps"].calculate_score(preds_imgs, ref_imgs)

        print("ssim score = {}".format(ssim_score))
        print("psnr score = {}".format(psnr_score))
        print("lps score = {}".format(lps_score))

        self.assertEqual(ssim_score, 1.0)
        self.assertEqual(psnr_score, np.inf)
        self.assertEqual(lps_score, 0.0)

    def test_02_unpaired_metrics(self):
        bs = 5
        image_size = 512
        preds_imgs = np.random.rand(bs, 3, image_size, image_size)
        preds_imgs *= 255
        preds_imgs = preds_imgs.astype(np.uint8)

        ref_imgs = np.random.rand(bs, 3, image_size, image_size)
        ref_imgs *= 255
        ref_imgs = ref_imgs.astype(np.uint8)

        inception_score = self.unpaired_metric_dict["is"].calculate_score(preds_imgs)
        fid_score = self.unpaired_metric_dict["fid"].calculate_score(preds_imgs, ref_imgs)
        os_cs_reid = self.unpaired_metric_dict["OS-CS-reid"].calculate_score(preds_imgs, ref_imgs)
        pcb_cs_reid = self.unpaired_metric_dict["PCB-CS-reid"].calculate_score(preds_imgs, ref_imgs)
        os_freid = self.unpaired_metric_dict["OS-freid"].calculate_score(preds_imgs, ref_imgs)
        pcb_freid = self.unpaired_metric_dict["PCB-freid"].calculate_score(preds_imgs, ref_imgs)

        print("inception score = {}".format(inception_score))
        print("fid score = {}".format(fid_score))
        print("OS-Cosine Similarity = {}".format(os_cs_reid))
        print("PCB-Cosine Similarity = {}".format(pcb_cs_reid))
        print("OS-freid = {}".format(os_freid))
        print("PCB-freid = {}".format(pcb_freid))


if __name__ == '__main__':
    unittest.main()
