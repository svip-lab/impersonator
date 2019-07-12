from .base_options import BaseOptions


class EvalOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)

        self._parser.add_argument('--n_threads_train', default=4, type=int, help='# threads for loading data')
        self._parser.add_argument('--is_both', action='store_true',
                                  help='whether to use generate background for both source and target image.')
        self._parser.add_argument('--visual', action='store_true', help='using visualizer or not.')
        self._parser.add_argument('--out_dir', type=str, default='', help='output dir')
        self._parser.add_argument('--ip', type=str, default='http://10.19.129.77', help='visdom ip')
        self._parser.add_argument('--port', type=int, default=10086, help='visdom port')
        self.is_train = False
