from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self._parser.add_argument('--output_dir', type=str, default='./outputs/results/',
                                  help='output directory to save the results')
        self._parser.add_argument('--src_path', type=str, default='', help='source image path')
        self._parser.add_argument('--tgt_path', type=str, default='', help='target image path')
        self._parser.add_argument('--pri_path', type=str, default='./assets/samples/A_priors/imgs',
                                  help='prior image path')

        self._parser.add_argument('--load_path', type=str,
                                  default='./outputs/checkpoints/lwb_imper_fashion_place/net_epoch_30_id_G.pth',
                                  help='pretrained model path')

        self._parser.add_argument('--bg_model', type=str,
                                  default='./outputs/checkpoints/deepfillv2/net_epoch_50_id_G.pth',
                                  help='if it is `ORIGINAL`, it will use the '
                                       'original BGNet of the generator of LiquidWarping GAN, '
                                       'otherwise, set it as `./outputs/checkpoints/deepfillv2/net_epoch_50_id_G.pth`, '
                                       'and it will use a pretrained deepfillv2 background inpaintor (default).')

        self._parser.add_argument('--bg_ks', default=13, type=int, help='dilate kernel size of background mask.')
        self._parser.add_argument('--ft_ks', default=3, type=int, help='dilate kernel size of front mask.')
        self._parser.add_argument('--only_vis', action="store_true", default=False, help='only visible or not')
        self._parser.add_argument('--has_detector', action='store_true', help='use mask rcnn or not')
        self._parser.add_argument('--body_seg', action="store_true", default=False,
                                  help='use the body segmentation estimated by mask rcnn.')
        self._parser.add_argument('--front_warp', action="store_true", default=False, help='front warp or not')
        self._parser.add_argument('--post_tune', action="store_true", default=False, help='post tune or not')

        # Human motion imitation
        self._parser.add_argument('--cam_strategy', type=str, default='smooth', choices=['smooth', 'source', 'copy'],
                                  help='the strategy to control the camera pameters (s, x, y) '
                                       'betwwen the source and reference image.')

        # Human appearance transfer
        self._parser.add_argument('--swap_part', type=str, default='body', help='part to swap')

        # Novel view synthesis
        self._parser.add_argument('--T_pose', action="store_true", default=False, help='view as T pose or not.')
        self._parser.add_argument('--view_params', type=str, default='R=0,90,0/t=0,0,0', help='params of novel view.')

        # visualizer
        self._parser.add_argument('--ip', type=str, default='', help='visdom ip')
        self._parser.add_argument('--port', type=int, default=31100, help='visdom port')

        # save results or not
        self._parser.add_argument('--save_res', action='store_true', default=False,
                                  help='save images or not, if true, the results are saved in `${output_dir}/preds`.')

        self.is_train = False
