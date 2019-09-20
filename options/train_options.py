from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)

        # use place dataset if need
        self._parser.add_argument('--place_dir', type=str, default='/p300/places365_standard', help='place folder')
        self._parser.add_argument('--place_bs', type=int, default=4, help='input batch size of place dataset')

        # use deep fashion dataset if need
        self._parser.add_argument('--fashion_dir', type=str, default='/public/deep_fashion/intrinsic', help='place folder')
        self._parser.add_argument('--fashion_bs', type=int, default=4, help='input batch size of fashion dataset')

        self._parser.add_argument('--intervals', type=int, default=10, help='the interval between frames.')
        self._parser.add_argument('--n_threads_train', default=4, type=int, help='# threads for loading data')
        self._parser.add_argument('--num_iters_validate', default=1, type=int, help='# batches to use when validating')
        self._parser.add_argument('--print_freq_s', type=int, default=60, help='frequency of showing training results on console')
        self._parser.add_argument('--display_freq_s', type=int, default=300, help='frequency [s] of showing training results on screen')
        self._parser.add_argument('--save_latest_freq_s', type=int, default=3600, help='frequency of saving the latest results')

        self._parser.add_argument('--use_vgg', action='store_true', help='whether to use VGG loss or L1 loss, if true use VGG, other use L1, default is L1')
        self._parser.add_argument('--use_style', action='store_true', help='whether to use style loss or not')
        self._parser.add_argument('--use_face', action='store_true', help='whether to use face loss or not')
        self._parser.add_argument('--mask_bce', action='store_true', help='whether to use CrossEntropyLoss or L1 loss in mask or not.')
        self._parser.add_argument('--nepochs_no_decay', type=int, default=10, help='# of epochs at starting learning rate')
        self._parser.add_argument('--nepochs_decay', type=int, default=20, help='# of epochs to linearly decay learning rate to zero')

        self._parser.add_argument('--train_G_every_n_iterations', type=int, default=1, help='train G every n interations')
        self._parser.add_argument('--final_lr', type=float, default=0.000002, help='final learning rate')
        self._parser.add_argument('--lr_G', type=float, default=0.0002, help='initial learning rate for G adam')
        self._parser.add_argument('--G_adam_b1', type=float, default=0.5, help='beta1 for G adam')
        self._parser.add_argument('--G_adam_b2', type=float, default=0.999, help='beta2 for G adam')
        self._parser.add_argument('--lr_D', type=float, default=0.0002, help='initial learning rate for D adam')
        self._parser.add_argument('--D_adam_b1', type=float, default=0.5, help='beta1 for D adam')
        self._parser.add_argument('--D_adam_b2', type=float, default=0.999, help='beta2 for D adam')
        self._parser.add_argument('--lambda_D_prob', type=float, default=1, help='lambda for real/fake discriminator loss')
        self._parser.add_argument('--lambda_lp', type=float, default=10, help='lambda lp loss')
        self._parser.add_argument('--lambda_vgg', type=float, default=10, help='lambda vgg loss')
        self._parser.add_argument('--lambda_style', type=float, default=5, help='lambda style loss')
        self._parser.add_argument('--lambda_face', type=float, default=1, help='lambda face loss')
        self._parser.add_argument('--lambda_mask', type=float, default=0.1, help='lambda mask loss')
        self._parser.add_argument('--lambda_mask_smooth', type=float, default=1e-5, help='lambda mask smooth loss')

        self.is_train = True
