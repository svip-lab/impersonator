import matplotlib.pyplot as plt
import os
import cv2
import numpy as np


from utils.visualizer.demo_visualizer import MotionImitationVisualizer


def load_img(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


class PltVisualizer(object):

    def __init__(self):
        default = np.zeros((256, 256, 3), dtype=np.uint8)

        fig = plt.figure(1)
        self.src = fig.add_subplot(141)
        self.src_img = self.src.imshow(default)
        plt.title('src')

        self.ref = fig.add_subplot(142)
        self.ref_img = self.ref.imshow(default)
        plt.title('ref')

        self.mit = fig.add_subplot(143)
        self.mit_img = self.mit.imshow(default)
        plt.title('mit')

        self.our = fig.add_subplot(144)
        self.our_img = self.our.imshow(default)
        plt.title('ours')

    def source(self, src_path):
        src = cv2.imread(src_path)
        src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        self.src.imshow(src)

    def show(self, ref_path, mit_path, our_path):
        ref = cv2.imread(ref_path)
        mit = cv2.imread(mit_path)
        our = cv2.imread(our_path)

        ref = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)
        mit = cv2.cvtColor(mit, cv2.COLOR_BGR2RGB)
        our = cv2.cvtColor(our, cv2.COLOR_BGR2RGB)

        self.ref.imshow(ref)
        self.ref.set_title('ref')

        self.mit.imshow(mit)
        self.mit.set_title('mit')

        self.our.imshow(our)
        self.our.set_title('our')

        plt.draw()
        plt.pause(0.1)
        self.ref.cla()
        self.mit.cla()
        self.our.cla()


def show(visualizer,
         mit_dir='/home/piaozx/liuwen/p300/results/pG2',
         our_dir='/home/piaozx/liuwen/p300/results/impersonator_02_29'):

    video_names = os.listdir(mit_dir)[::-1]
    for video_name in video_names:
        src_names = [str(name.split('.')[0]) for name in os.listdir(os.path.join(mit_dir, video_name))]

        for src_name in src_names:
            src_img_path = os.path.join(mit_dir, video_name, src_name + '.jpg')
            imgs_mit_dir = os.path.join(mit_dir, video_name, src_name)
            imgs_our_dir = os.path.join(our_dir, video_name, src_name)

            visualizer.vis_named_img('src', load_img(src_img_path)[None], transpose=True, normalize=True)

            for v_name in os.listdir(imgs_mit_dir):
                v_mit_dir = os.path.join(imgs_mit_dir, v_name)
                v_our_dir = os.path.join(imgs_our_dir, v_name)

                images = [str(name.split('_')[-1]) for name in os.listdir(v_mit_dir) if 'pred' in name]

                for image in images:
                    gt_path = os.path.join(v_mit_dir, 'gt_' + image)
                    mit_pred_path = os.path.join(v_mit_dir, 'pred_' + image)
                    our_pred_path = os.path.join(v_our_dir, 'pred_' + image)

                    gt = load_img(gt_path)
                    mit = load_img(mit_pred_path)
                    our = load_img(our_pred_path)

                    imgs = np.stack([gt, mit, our], axis=0)

                    visualizer.vis_named_img('out', imgs, transpose=True, normalize=True)

                    print(v_name, image)


if __name__ == '__main__':
    visualizer = MotionImitationVisualizer(env='debug', ip='http://10.19.125.13', port=10087)

    show(visualizer)