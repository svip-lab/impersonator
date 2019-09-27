import argparse
import os
from utils import util


class BaseOptions(object):
    def __init__(self):
        self._parser = argparse.ArgumentParser()
        self._initialized = False

    def initialize(self):
        self._parser.add_argument('--checkpoints_dir', type=str, default='./outputs/checkpoints/',
                                  help='models are saved here')

        self._parser.add_argument('--data_dir', type=str, default='/p300/datasets/iPER', help='path to dataset')
        self._parser.add_argument('--dataset_mode', type=str, default='iPER', help='chooses dataset to be used')
        self._parser.add_argument('--train_ids_file', type=str, default='train.txt', help='file containing train ids')
        self._parser.add_argument('--test_ids_file', type=str, default='val.txt', help='file containing test ids')
        self._parser.add_argument('--images_folder', type=str, default='images_HD', help='images folder')
        self._parser.add_argument('--smpls_folder', type=str, default='smpls', help='smpls folder')

        self._parser.add_argument('--map_name', type=str, default='uv_seg', help='mapping function')
        self._parser.add_argument('--part_info', type=str, default='assets/pretrains/smpl_part_info.json',
                                  help='smpl part info path.')
        self._parser.add_argument('--uv_mapping', type=str, default='assets/pretrains/mapper.txt',
                                  help='uv mapping.')
        self._parser.add_argument('--hmr_model', type=str, default='assets/pretrains/hmr_tf2pt.pth',
                                  help='pretrained hmr model path.')
        self._parser.add_argument('--smpl_model', type=str, default='assets/pretrains/smpl_model.pkl',
                                  help='pretrained smpl model path.')
        self._parser.add_argument('--face_model', type=str, default='assets/pretrains/sphere20a_20171020.pth',
                                  help='pretrained face model path.')

        self._parser.add_argument('--load_epoch', type=int, default=-1,
                                  help='which epoch to load? set to -1 to use latest cached model')
        self._parser.add_argument('--batch_size', type=int, default=4, help='input batch size')
        self._parser.add_argument('--time_step', type=int, default=10, help='time step size')
        self._parser.add_argument('--tex_size', type=int, default=3, help='input tex size')
        self._parser.add_argument('--image_size', type=int, default=256, help='input image size')
        self._parser.add_argument('--repeat_num', type=int, default=6, help='number of residual blocks.')
        self._parser.add_argument('--cond_nc', type=int, default=3, help='# of conditions')
        self._parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self._parser.add_argument('--model', type=str, default='impersonator', help='model to run')
        self._parser.add_argument('--name', type=str, default='running',
                                  help='name of the experiment. It decides where to store samples and models')
        self._parser.add_argument('--gen_name', type=str, default='impersonator',
                                  help='chooses generator to be used, resnet or unet')
        self._parser.add_argument('--norm_type', type=str, default='instance',
                                  help='choose use what norm layer in discriminator')
        self._parser.add_argument('--n_threads_test', default=2, type=int, help='# threads for loading data')
        self._parser.add_argument('--serial_batches', action='store_true',
                                  help='if true, takes images in order to make batches, otherwise takes them randomly')
        self._parser.add_argument('--do_saturate_mask', action="store_true",
                                  default=False, help='do use mask_fake for mask_cyc')
        self._parser.add_argument('--bg_replace', action="store_true",
                                  default=False, help='replace original pixels or not')
        self._parser.add_argument('--debug', action="store_true",
                                  default=False, help='debug or not')
        self._initialized = True

    def parse(self):
        if not self._initialized:
            self.initialize()
        self._opt = self._parser.parse_args()

        # set is train or set
        self._opt.is_train = self.is_train

        # set and check load_epoch
        self._set_and_check_load_epoch()

        # get and set gpus
        self._get_set_gpus()

        args = vars(self._opt)

        # print in terminal args
        self._print(args)

        # save args to file
        self._save(args)

        return self._opt

    def _set_and_check_load_epoch(self):
        models_dir = os.path.join(self._opt.checkpoints_dir, self._opt.name)
        if os.path.exists(models_dir):
            if self._opt.load_epoch == -1:
                load_epoch = 0
                for file in os.listdir(models_dir):
                    if file.startswith("net_epoch_"):
                        load_epoch = max(load_epoch, int(file.split('_')[2]))
                self._opt.load_epoch = load_epoch
            else:
                found = False
                for file in os.listdir(models_dir):
                    if file.startswith("net_epoch_"):
                        found = int(file.split('_')[2]) == self._opt.load_epoch
                        if found: break
                assert found, 'Model for epoch %i not found' % self._opt.load_epoch
        else:
            assert self._opt.load_epoch < 1, 'Model for epoch %i not found' % self._opt.load_epoch
            self._opt.load_epoch = 0

    def _get_set_gpus(self):
        os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"

        if len(self._opt.gpu_ids) > 0:
            os.environ['CUDA_VISIBLE_DEVICES'] = self._opt.gpu_ids
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    def _print(self, args):
        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

    def _save(self, args):
        expr_dir = os.path.join(self._opt.checkpoints_dir, self._opt.name)
        print(expr_dir)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt_%s.txt' % ('train' if self.is_train else 'test'))
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
