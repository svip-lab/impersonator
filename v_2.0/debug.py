import time
from options.train_options import TrainOptions
from data.custom_dataset_data_loader import CustomDatasetDataLoader
from models.models import ModelsFactory
from utils.visualizer.demo_visualizer import MotionImitationVisualizer

# --ip http://10.19.126.34 --port 10087
ip = 'http://10.19.126.34'
port = 10087
# ip = 'http://10.10.10.100'
# port = 31100


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

    def collect_samples(self):
        import numpy as np
        from utils.util import write_pickle_file
        data = {
            'src_imgs': [],
            'ref_imgs': [],
            'dst_imgs': [],
            'obj_imgs': [],
            'obj_masks': []
        }
        for i in range(50):
            src_imgs, ref_imgs, dst_imgs, obj_imgs, obj_masks = self._model.debug_get_data(self._visualizer)
            data['src_imgs'].append(src_imgs)
            data['ref_imgs'].append(ref_imgs)
            data['dst_imgs'].append(dst_imgs)
            data['obj_imgs'].append(obj_imgs)
            data['obj_masks'].append(obj_masks)
            print(i)

        for key in data:
            data[key] = np.concatenate(data[key])
            print(key, data[key].shape)

        write_pickle_file('samples.pkl', data)

    # def _train_epoch(self, i_epoch):
    #     epoch_iter = 0
    #     self._model.set_train()
    #     for i_train_batch, train_batch in enumerate(self._dataset_train):
    #         iter_start_time = time.time()
    #
    #         # display flags
    #         do_visuals = self._last_display_time is None or time.time() - self._last_display_time > self._opt.display_freq_s
    #         do_print_terminal = time.time() - self._last_print_time > self._opt.print_freq_s or do_visuals
    #
    #         # train model
    #         self._model.set_input(train_batch)
    #         print('set input')
    #
    #         # debug
    #         # self._model.debug(self._visualizer)
    #         self.collect_samples()
    #         self._model.debug_wrap(self._visualizer)

    def _train_epoch(self, i_epoch):
        import numpy as np
        from utils.util import write_pickle_file
        data = {
            'src_imgs': [],
            'ref_imgs': [],
            'dst_imgs': [],
            'obj_imgs': [],
            'obj_masks': []
        }
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

            src_imgs, ref_imgs, dst_imgs, obj_imgs, obj_masks = self._model.debug_get_data(self._visualizer)
            data['src_imgs'].append(src_imgs)
            data['ref_imgs'].append(ref_imgs)
            data['dst_imgs'].append(dst_imgs)
            data['obj_imgs'].append(obj_imgs)
            data['obj_masks'].append(obj_masks)
            print(i_train_batch)

            if i_train_batch >= 40:
                break

        for key in data:
            data[key] = np.concatenate(data[key])
            print(key, data[key].shape)

        write_pickle_file('samples.pkl', data)


if __name__ == "__main__":
    Train()
