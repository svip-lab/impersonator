import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt


def load_pickle_file(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    return data


def directly_paste(obj, dst, mask):
    """
    :param obj:
    :param dst:
    :param mask:
    :return:
    """
    out = obj * mask + (1 - mask) * dst
    out = (out + 1) / 2.0
    return out


def seamless_paste(obj, dst, mask):
    """
    :param obj:
    :param dst:
    :param mask:
    :return:
    """
    mask = mask * 255
    mask = mask.astype(np.uint8)

    obj = (obj + 1) / 2 * 255
    obj = obj.astype(np.uint8)

    # The location of the center of the src in the dst
    # width, height, channels = dst.shape
    # center = (height // 2, width // 2)
    y, x, _ = np.where(mask != 0)
    #
    center = (int(np.mean(x)), int(np.mean(y)))

    dst = (dst + 1) / 2 * 255
    dst = dst.astype(np.uint8)
    # Seamlessly clone obj into dst and put the results in output
    normal_clone = cv2.seamlessClone(obj, dst, mask, center, cv2.NORMAL_CLONE)
    # normal_clone = cv2.seamlessClone(obj, dst, mask, center, cv2.MIXED_CLONE)
    normal_clone = normal_clone.astype(np.float32)
    normal_clone /= 255.0

    mixed_clone = cv2.seamlessClone(obj, dst, mask, center, cv2.MIXED_CLONE)
    mixed_clone = mixed_clone.astype(np.float32)
    mixed_clone /= 255.0

    return normal_clone, mixed_clone


class PltVisualizer(object):

    def __init__(self):

        fig = plt.figure(1)
        self.dst = fig.add_subplot(151)
        self.obj = fig.add_subplot(152)
        self.seam_normal = fig.add_subplot(153)
        self.seam_mixed = fig.add_subplot(154)
        self.copy = fig.add_subplot(155)

    def show(self, dst, obj, seam_normal, seam_mixed, copy):
        self.dst.imshow(dst)
        self.dst.set_title('dst')
        self.dst.axis('off')

        self.obj.imshow(obj)
        self.obj.set_title('obj')
        self.obj.axis('off')

        self.seam_normal.imshow(seam_normal)
        self.seam_normal.set_title('seam_normal')
        self.seam_normal.axis('off')

        self.seam_mixed.imshow(seam_mixed)
        self.seam_mixed.set_title('seam_mixed')
        self.seam_mixed.axis('off')

        self.copy.imshow(copy)
        self.copy.set_title('copy')
        self.copy.axis('off')

        plt.draw()
        plt.pause(10)
        self.dst.cla()
        self.obj.cla()
        self.seam_normal.cla()
        self.seam_mixed.cla()
        self.copy.cla()


if __name__ == '__main__':
    load_samples = load_pickle_file('samples.pkl')

    src_imgs = load_samples['src_imgs']     # (50, 3, 256, 256), rgb [-1, 1]
    ref_imgs = load_samples['ref_imgs']     # (50, 3, 256, 256), rgb [-1, 1]
    dst_imgs = load_samples['dst_imgs']     # (50, 3, 256, 256), rgb [-1, 1]
    obj_imgs = load_samples['obj_imgs']     # (50, 3, 256, 256), rgb [-1, 1]
    obj_masks = load_samples['obj_masks']   # (50, 1, 256, 256), 0 or 1

    num_samples = src_imgs.shape[0]

    # define visualizer
    visualizer = PltVisualizer()
    for i in range(num_samples):
        dst = np.transpose(dst_imgs[i], axes=(1, 2, 0))     # (256, 256, 3)
        obj = np.transpose(obj_imgs[i], axes=(1, 2, 0))     # (256, 256, 3)
        mask = obj_masks[i, 0][:, :, np.newaxis]            # (256, 256, 1)

        # 1. seamless paste, if mask has more than 50 pixels, other wise normalize [-1, 1] to [0, 1]
        if np.sum(mask) > 50:
            seam_normal, seam_mixed = seamless_paste(obj, dst, mask)
        else:
            out = (dst + 1) / 2.0
            seam_normal, seam_mixed = out, out

        # 2. directly paste
        copy_out = directly_paste(obj, dst, mask)

        dst = (dst + 1) / 2
        obj = (obj + 1) / 2

        print(i, '10s for the next sample.')
        visualizer.show(dst, obj, seam_normal, seam_mixed, copy_out)