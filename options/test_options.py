from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self._parser.add_argument('--output_dir', type=str, default='/p300/poseGANs', help='output path')
        self._parser.add_argument('--load_path', type=str, default='', help='pretrained model path')
        self._parser.add_argument('--src_path', type=str, default='', help='source image path')
        self._parser.add_argument('--tgt_path', type=str, default='', help='target image path')
        self._parser.add_argument('--ref_path', type=str, default='', help='reference image path')
        self._parser.add_argument('--pri_path', type=str, default='/root/poseGANs/meta_train/samples/ref_imgs/1',
                                  help='prior image path')

        self._parser.add_argument('--bg_model', type=str, default='', help='pretrain bgnet if use.')
        self._parser.add_argument('--has_detector', action='store_true', help='use mask rcnn or not')
        self._parser.add_argument('--bg_ks', default=13, type=int, help='dilate kernel size.')
        self._parser.add_argument('--ft_ks', default=3, type=int, help='dilate kernel size.')
        self._parser.add_argument('--only_vis', action="store_true", default=False, help='only visible or not')
        self._parser.add_argument('--front_warp', action="store_true", default=False, help='front warp or not')
        self._parser.add_argument('--post_tune', action="store_true", default=False, help='post tune or not')

        # Motion transfer
        self._parser.add_argument('--cam_strategy', type=str, default='smooth',
                                  choices=['smooth', 'source', 'copy'],
                                  help='the flag to copy cam or not.')

        # Human appearance transfer
        self._parser.add_argument('--swap_part', type=str, default='upper_body', help='part to swap')

        # Novel view synthesis
        self._parser.add_argument('--view_params', type=str, default='R=0,90,0/t=0,0,0', help='params of novel view.')

        # visualizer
        self._parser.add_argument('--visual', action='store_true', help='using visualizer or not.')
        self._parser.add_argument('--ip', type=str, default='http://10.10.10.100', help='visdom ip')
        self._parser.add_argument('--port', type=int, default=31100, help='visdom port')

        self.is_train = False
