import numpy as np
import os
import time
from . import util
from tensorboardX import SummaryWriter
import ipdb


class TBVisualizer(object):
    def __init__(self, opt):
        self._opt = opt
        self._save_path = os.path.join(opt.checkpoints_dir, opt.name)

        self._log_path = os.path.join(self._save_path, 'loss_log2.txt')
        self._tb_path = os.path.join(self._save_path, 'summary.json')
        self._writer = SummaryWriter(self._save_path)

        with open(self._log_path, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def __del__(self):
        self._writer.close()

    def display_current_results(self, visuals, it, is_train, save_visuals=False):
        for label, image_numpy in visuals.items():
            # ipdb.set_trace()
            sum_name = '{}/{}'.format('Train' if is_train else 'Test', label)
            self._writer.add_image(sum_name, image_numpy, it)

            if save_visuals:
                util.save_image(image_numpy,
                                os.path.join(self._opt.checkpoints_dir, self._opt.name,
                                             'event_imgs', sum_name, '%08d.png' % it))

        self._writer.export_scalars_to_json(self._tb_path)

    def plot_scalars(self, scalars, it, is_train):
        for label, scalar in scalars.items():
            sum_name = '{}/{}'.format('Train' if is_train else 'Test', label)
            self._writer.add_scalar(sum_name, scalar, it)

    def print_current_train_errors(self, epoch, i, iters_per_epoch, errors, t, visuals_were_stored):
        log_time = time.strftime("[%d/%m/%Y %H:%M:%S]")
        visuals_info = "v" if visuals_were_stored else ""
        message = '%s (T%s, epoch: %d, it: %d/%d, t/smpl: %.3fs)\n' % (log_time, visuals_info, epoch, i, iters_per_epoch, t)
        for k, v in errors.items():
            msg = '\t%s:%.3f\n' % (k, v)
            message += msg
        print(message)
        with open(self._log_path, "a") as log_file:
            log_file.write('%s\n' % message)

    def print_current_validate_errors(self, epoch, errors, t):
        log_time = time.strftime("[%d/%m/%Y %H:%M:%S]")
        message = '%s (V, epoch: %d, time_to_val: %ds)\n' % (log_time, epoch, t)
        for k, v in errors.items():
            message += '\t%s:%.3f\n' % (k, v)

        print(message)
        with open(self._log_path, "a") as log_file:
            log_file.write('%s\n' % message)

    def save_images(self, visuals):
        for label, image_numpy in visuals.items():
            image_name = '%s.png' % label
            save_path = os.path.join(self._save_path, "samples", image_name)
            util.save_image(image_numpy, save_path)


if __name__ == '__main__':
    from options.train_options import TrainOptions
    from data.imper_dataset import SeqMIDataset
    from utils.util import tensor2im

    opts = TrainOptions().parse()

    pair_dataset = SeqMIDataset(opts, is_for_train=True)
    tb_visualizer = TBVisualizer(opts)

    for i in range(0, 100):
        sample = pair_dataset[i]
        images = sample['images']
        print(images.shape)
        visuals = {
            'images': tensor2im(sample['images'])
        }

        scalars = {
            'loss': np.random.rand()
        }

        tb_visualizer.display_current_results(visuals, i, is_train=True)
        tb_visualizer.plot_scalars(scalars, i, is_train=True)

        time.sleep(1)
