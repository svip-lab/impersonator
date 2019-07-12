import time
from options.train_options import TrainOptions
from data.custom_dataset_data_loader import CustomDatasetDataLoader
from models.models import ModelsFactory
from utils.demo_visualizer import MotionImitationVisualizer
from collections import OrderedDict
import os

ip = 'http://10.10.10.100'
port = 31100


class Train(object):
    def __init__(self):
        self._opt = TrainOptions().parse()
        data_loader_train = CustomDatasetDataLoader(self._opt, is_for_train=True)
        data_loader_test = CustomDatasetDataLoader(self._opt, is_for_train=False)

        self._dataset_train = data_loader_train.load_data()
        self._dataset_test = data_loader_test.load_data()

        self._dataset_train_size = len(data_loader_train)
        self._dataset_test_size = len(data_loader_test)
        print('#train video clips = %d' % self._dataset_train_size)
        print('#test video clips = %d' % self._dataset_test_size)

        self._model = ModelsFactory.get_by_name(self._opt.model, self._opt)
        self._visualizer = MotionImitationVisualizer(env='debug', ip=ip, port=port)

        self._train()

    def _train(self):
        self._total_steps = self._opt.load_epoch * self._dataset_train_size
        self._iters_per_epoch = self._dataset_train_size / self._opt.batch_size
        self._last_display_time = None
        self._last_save_latest_time = None
        self._last_print_time = time.time()

        for i_epoch in range(self._opt.load_epoch + 1, self._opt.nepochs_no_decay + self._opt.nepochs_decay + 1):
            epoch_start_time = time.time()

            # train epoch
            self._train_epoch(i_epoch)

            # print epoch info
            time_epoch = time.time() - epoch_start_time
            print('End of epoch %d / %d \t Time Taken: %d sec (%d min or %d h)' %
                  (i_epoch, self._opt.nepochs_no_decay + self._opt.nepochs_decay, time_epoch,
                   time_epoch / 60, time_epoch / 3600))

            # update learning rate
            if i_epoch > self._opt.nepochs_no_decay:
                self._model.update_learning_rate()

    def _train_epoch(self, i_epoch):
        epoch_iter = 0
        self._model.set_train()
        for i_train_batch, train_batch in enumerate(self._dataset_train):
            iter_start_time = time.time()

            # display flags
            do_visuals = self._last_display_time is None or time.time() - self._last_display_time > self._opt.display_freq_s
            do_print_terminal = time.time() - self._last_print_time > self._opt.print_freq_s or do_visuals

            # train model
            self._model.set_input(train_batch)
            print('set input')

            # debug
            self._model.debug_fim_transfer(self._visualizer)
            # self._model.debug_binary_fim(self._visualizer)
            # self._model.debug_face(self._visualizer)
            # self._model.debug(self._visualizer)
            # self._model.debug_front_face(self._visualizer)

            # train_generator = ((i_train_batch+1) % self._opt.train_G_every_n_iterations == 0) or do_visuals
            # self._model.optimize_parameters(keep_data_for_visuals=do_visuals, train_generator=train_generator)
            #
            # # update epoch info
            # self._total_steps += self._opt.batch_size
            # epoch_iter += self._opt.batch_size


if __name__ == "__main__":
    Train()
