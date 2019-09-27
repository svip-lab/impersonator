import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from .models import BaseModel
from networks.networks import NetworksFactory, HumanModelRecovery
from utils.nmr import SMPLRenderer
from utils.detectors import PersonMaskRCNNDetector
import utils.cv_utils as cv_utils
import utils.util as util


class Viewer(BaseModel):

    def __init__(self, opt):
        super(Viewer, self).__init__(opt)
        self._name = 'Viewer'

        self._create_networks()

        # prefetch variables
        self.src_info = None
        self.tsf_info = None
        self.first_cam = None

    def _create_networks(self):
        # 0. create generator
        self.generator = self._create_generator().cuda()

        # 0. create bgnet
        if self._opt.bg_model != 'ORIGINAL':
            self.bgnet = self._create_bgnet().cuda()
        else:
            self.bgnet = self.generator.bg_model

        # 2. create hmr
        self.hmr = self._create_hmr().cuda()

        # 3. create render
        self.render = SMPLRenderer(image_size=self._opt.image_size, tex_size=self._opt.tex_size,
                                   has_front=self._opt.front_warp, fill_back=False).cuda()
        # 4. pre-processor
        if self._opt.has_detector:
            self.detector = PersonMaskRCNNDetector(ks=self._opt.bg_ks, threshold=0.5, to_gpu=True)
        else:
            self.detector = None

    def _create_bgnet(self):
        net = NetworksFactory.get_by_name('deepfillv2', c_dim=4)
        self._load_params(net, self._opt.bg_model, need_module=False)
        net.eval()
        return net

    def _create_generator(self):
        net = NetworksFactory.get_by_name(self._opt.gen_name, bg_dim=4, src_dim=3+self._G_cond_nc,
                                          tsf_dim=3+self._G_cond_nc, repeat_num=self._opt.repeat_num)

        if self._opt.load_path:
            self._load_params(net, self._opt.load_path)
        elif self._opt.load_epoch > 0:
            self._load_network(net, 'G', self._opt.load_epoch)
        else:
            raise ValueError('load_path {} is empty and load_epoch {} is 0'.format(
                self._opt.load_path, self._opt.load_epoch))

        net.eval()
        return net

    def _create_hmr(self):
        hmr = HumanModelRecovery(self._opt.smpl_model)
        saved_data = torch.load(self._opt.hmr_model)
        hmr.load_state_dict(saved_data)
        hmr.eval()
        return hmr

    def visualize(self, *args, **kwargs):
        visualizer = args[0]
        if visualizer is not None:
            for key, value in kwargs.items():
                visualizer.vis_named_img(key, value)

    @torch.no_grad()
    def personalize(self, src_path, src_smpl=None, output_path='', visualizer=None):

        ori_img = cv_utils.read_cv2_img(src_path)

        # resize image and convert the color space from [0, 255] to [-1, 1]
        img = cv_utils.transform_img(ori_img, self._opt.image_size, transpose=True) * 2 - 1.0
        img = torch.tensor(img, dtype=torch.float32).cuda()[None, ...]

        if src_smpl is None:
            img_hmr = cv_utils.transform_img(ori_img, 224, transpose=True) * 2 - 1.0
            img_hmr = torch.tensor(img_hmr, dtype=torch.float32).cuda()[None, ...]
            src_smpl = self.hmr(img_hmr)
        else:
            src_smpl = torch.tensor(src_smpl, dtype=torch.float32).cuda()[None, ...]

        # source process, {'theta', 'cam', 'pose', 'shape', 'verts', 'j2d', 'j3d'}
        src_info = self.hmr.get_details(src_smpl)
        src_f2verts, src_fim, src_wim = self.render.render_fim_wim(src_info['cam'], src_info['verts'])
        # src_f2pts = src_f2verts[:, :, :, 0:2]
        src_info['fim'] = src_fim
        src_info['wim'] = src_wim
        src_info['cond'], _ = self.render.encode_fim(src_info['cam'], src_info['verts'], fim=src_fim, transpose=True)
        src_info['f2verts'] = src_f2verts
        src_info['p2verts'] = src_f2verts[:, :, :, 0:2]
        src_info['p2verts'][:, :, :, 1] *= -1

        if self._opt.only_vis:
            src_info['p2verts'] = self.render.get_vis_f2pts(src_info['p2verts'], src_fim)
        # add image to source info
        src_info['img'] = img
        src_info['image'] = ori_img

        # 2. process the src inputs
        if self.detector is not None:
            bbox, body_mask = self.detector.inference(img[0])
            bg_mask = 1 - body_mask
        else:
            bg_mask = util.morph(src_info['cond'][:, -1:, :, :], ks=self._opt.bg_ks, mode='erode')
            body_mask = 1 - bg_mask

        if self._opt.bg_model != 'ORIGINAL':
            src_info['bg'] = self.bgnet(img, masks=body_mask, only_x=True)
        else:
            incomp_img = img * bg_mask
            bg_inputs = torch.cat([incomp_img, bg_mask], dim=1)
            img_bg = self.bgnet(bg_inputs)
            src_info['bg'] = bg_inputs[:, 0:3] + img_bg * bg_inputs[:, -1:]

        ft_mask = 1 - util.morph(src_info['cond'][:, -1:, :, :], ks=self._opt.ft_ks, mode='erode')
        src_inputs = torch.cat([img * ft_mask, src_info['cond']], dim=1)

        src_info['feats'] = self.generator.encode_src(src_inputs)

        self.src_info = src_info

        if visualizer is not None:
            visualizer.vis_named_img('src', img)
            visualizer.vis_named_img('bg', src_info['bg'])

        if output_path:
            cv_utils.save_cv2_img(src_info['image'], output_path, image_size=self._opt.image_size)

    @torch.no_grad()
    def _extract_smpls(self, input_file):
        img = cv_utils.read_cv2_img(input_file)
        img = cv_utils.transform_img(img, image_size=224) * 2 - 1.0  # hmr receive [-1, 1]
        img = img.transpose((2, 0, 1))
        img = torch.tensor(img, dtype=torch.float32).cuda()[None, ...]
        theta = self.hmr(img)[-1]

        return theta

    @torch.no_grad()
    def inference(self, tgt_paths, tgt_smpls=None, cam_strategy='smooth', output_dir='', visualizer=None, verbose=True):
        length = len(tgt_paths)

        outputs = []
        bg_img = self.src_info['bg']
        src_encoder_outs, src_resnet_outs = self.src_info['feats']

        process_bar = tqdm(range(length)) if verbose else range(length)
        for t in process_bar:
            tgt_path = tgt_paths[t]
            tgt_smpl = tgt_smpls[t] if tgt_smpls is not None else None

            tsf_inputs = self.transfer_params(tgt_path, tgt_smpl, cam_strategy, t=t)

            tsf_color, tsf_mask = self.generator.inference(src_encoder_outs, src_resnet_outs,
                                                           tsf_inputs, self.tsf_info['T'])
            preds = tsf_mask * bg_img + (1 - tsf_mask) * tsf_color

            if self._opt.front_warp:
                preds = self.warp_front(preds, self.tsf_info['tsf_img'], self.tsf_info['fim'], tsf_mask)

            if visualizer is not None:
                gt = cv_utils.transform_img(self.tsf_info['image'], image_size=self._opt.image_size, transpose=True)
                visualizer.vis_named_img('pred_' + cam_strategy, preds)
                visualizer.vis_named_img('gt', gt[None, ...], denormalize=False)

            preds = preds[0].permute(1, 2, 0)
            preds = preds.cpu().numpy()
            outputs.append(preds)

            if output_dir:
                filename = os.path.split(tgt_path)[-1]

                cv_utils.save_cv2_img(preds, os.path.join(output_dir, 'pred_' + filename), normalize=True)
                cv_utils.save_cv2_img(self.tsf_info['image'], os.path.join(output_dir, 'gt_' + filename),
                                      image_size=self._opt.image_size)

        return outputs

    @torch.no_grad()
    def swap_smpl(self, src_cam, src_shape, tgt_smpl, cam_strategy='smooth'):
        tgt_cam = tgt_smpl[:, 0:3].contiguous()
        pose = tgt_smpl[:, 3:75].contiguous()

        # TODO, need more tricky ways
        if cam_strategy == 'smooth':

            cam = src_cam.clone()
            delta_xy = tgt_cam[:, 1:] - self.first_cam[:, 1:]
            cam[:, 1:] += delta_xy

        elif cam_strategy == 'source':
            cam = src_cam
        else:
            cam = tgt_cam

        tsf_smpl = torch.cat([cam, pose, src_shape], dim=1)

        return tsf_smpl

    @torch.no_grad()
    def transfer_params(self, tgt_path, tgt_smpl=None, cam_strategy='smooth', t=0):
        # get source info
        src_info = self.src_info

        ori_img = cv_utils.read_cv2_img(tgt_path)
        if tgt_smpl is None:
            img_hmr = cv_utils.transform_img(ori_img, 224, transpose=True) * 2 - 1.0
            img_hmr = torch.tensor(img_hmr, dtype=torch.float32).cuda()[None, ...]
            tgt_smpl = self.hmr(img_hmr)
        else:
            tgt_smpl = torch.tensor(tgt_smpl, dtype=torch.float32).cuda()[None, ...]

        if t == 0 and cam_strategy == 'smooth':
            self.first_cam = tgt_smpl[:, 0:3].clone()

        # get transfer smpl
        tsf_smpl = self.swap_smpl(src_info['cam'], src_info['shape'], tgt_smpl, cam_strategy=cam_strategy)
        # transfer process, {'theta', 'cam', 'pose', 'shape', 'verts', 'j2d', 'j3d'}
        tsf_info = self.hmr.get_details(tsf_smpl)

        tsf_f2verts, tsf_fim, tsf_wim = self.render.render_fim_wim(tsf_info['cam'], tsf_info['verts'])
        # src_f2pts = src_f2verts[:, :, :, 0:2]
        tsf_info['fim'] = tsf_fim
        tsf_info['wim'] = tsf_wim
        tsf_info['cond'], _ = self.render.encode_fim(tsf_info['cam'], tsf_info['verts'], fim=tsf_fim, transpose=True)
        # tsf_info['sil'] = util.morph((tsf_fim != -1).float(), ks=self._opt.ft_ks, mode='dilate')

        T = self.render.cal_bc_transform(src_info['p2verts'], tsf_fim, tsf_wim)
        tsf_img = F.grid_sample(src_info['img'], T)
        tsf_inputs = torch.cat([tsf_img, tsf_info['cond']], dim=1)

        # add target image to tsf info
        tsf_info['tsf_img'] = tsf_img
        tsf_info['image'] = ori_img
        tsf_info['T'] = T

        self.T = T
        self.tsf_info = tsf_info

        return tsf_inputs

    def warp_front(self, preds, tsf_img, fim, mask):
        front_mask = self.render.encode_front_fim(fim, transpose=True, front_fn=True)
        preds = (1 - front_mask) * preds + tsf_img * front_mask * (1 - mask)
        # preds = torch.clamp(preds + self.tsf_info['tsf_img'] * front_mask, -1, 1)
        return preds

    def rotate_trans(self, rt, t, X):
        R = cv_utils.euler2matrix(rt)    # (3 x 3)

        R = torch.FloatTensor(R)[None, :, :].cuda()
        t = torch.FloatTensor(t)[None, None, :].cuda()

        # (bs, Nv, 3) + (bs, 1, 3)
        return torch.bmm(X, R) + t

    def view(self, rt, t, visualizer=None, name='1'):

        src_info = self.src_info
        src_mesh = self.src_info['verts']
        tsf_mesh = self.rotate_trans(rt, t, src_mesh)

        tsf_f2verts, tsf_fim, tsf_wim = self.render.render_fim_wim(src_info['cam'], tsf_mesh)
        tsf_cond, _ = self.render.encode_fim(src_info['cam'], tsf_mesh, fim=tsf_fim, transpose=True)

        T = self.render.cal_bc_transform(src_info['p2verts'], tsf_fim, tsf_wim)
        tsf_img = F.grid_sample(src_info['img'], T)
        tsf_inputs = torch.cat([tsf_img, tsf_cond], dim=1)

        if not self._opt.bg_replace:
            bg = torch.zeros_like(src_info['bg'])
        else:
            bg = src_info['bg']
        preds, tsf_mask = self.forward(tsf_inputs, src_info['feats'], T, bg)

        if self._opt.front_warp:
            preds = self.warp_front(preds, tsf_img, tsf_fim, tsf_mask)

        if visualizer is not None:
            # self.render.set_ambient_light()
            # textures = self.render.debug_textures()[None, ...].cuda()
            # src_mesh, _ = self.render.render(src_info['cam'], src_X, textures, reverse_yz=True, get_fim=False)
            # tsf_mesh, _ = self.render.render(src_info['cam'], tsf_X, textures, reverse_yz=True, get_fim=False)

            visualizer.vis_named_img('src_img', src_info['img'])
            visualizer.vis_named_img('pred_' + name, preds)
            visualizer.vis_named_img('cond_' + name, tsf_cond)

        return preds

    def forward(self, tsf_inputs, feats, T, bg):
        # generate fake images
        src_encoder_outs, src_resnet_outs = feats

        tsf_color, tsf_mask = self.generator.inference(src_encoder_outs, src_resnet_outs, tsf_inputs, T)
        pred_imgs = tsf_mask * bg + (1 - tsf_mask) * tsf_color

        return pred_imgs, tsf_mask

    def post_personalize(self, out_dir, data_loader, visualizer, verbose=True):
        from networks.networks import FaceLoss

        bg_inpaint = self.src_info['bg']

        @torch.no_grad()
        def set_gen_inputs(sample):
            j2ds = sample['j2d'].cuda()  # (N, 4)
            T = sample['T'].cuda()  # (N, h, w, 2)
            T_cycle = sample['T_cycle'].cuda()  # (N, h, w, 2)
            src_inputs = sample['src_inputs'].cuda()  # (N, 6, h, w)
            tsf_inputs = sample['tsf_inputs'].cuda()  # (N, 6, h, w)
            src_fim = sample['src_fim'].cuda()
            tsf_fim = sample['tsf_fim'].cuda()
            init_preds = sample['preds'].cuda()
            images = sample['images']
            images = torch.cat([images[:, 0, ...], images[:, 1, ...]], dim=0).cuda()  # (2N, 3, h, w)
            pseudo_masks = sample['pseudo_masks']
            pseudo_masks = torch.cat([pseudo_masks[:, 0, ...], pseudo_masks[:, 1, ...]],
                                     dim=0).cuda()  # (2N, 1, h, w)

            return src_fim, tsf_fim, j2ds, T, T_cycle, \
                   src_inputs, tsf_inputs, images, init_preds, pseudo_masks

        def set_cycle_inputs(fake_tsf_imgs, src_inputs, tsf_inputs, T_cycle):
            # set cycle src inputs
            cycle_src_inputs = torch.cat([fake_tsf_imgs * tsf_inputs[:, -1:, ...], tsf_inputs[:, 3:]], dim=1)

            # set cycle tsf inputs
            cycle_tsf_img = F.grid_sample(fake_tsf_imgs, T_cycle)
            cycle_tsf_inputs = torch.cat([cycle_tsf_img, src_inputs[:, 3:]], dim=1)

            return cycle_src_inputs, cycle_tsf_inputs

        def warp(preds, tsf, fim, fake_tsf_mask):
            front_mask = self.render.encode_front_fim(fim, transpose=True)
            preds = (1 - front_mask) * preds + tsf * front_mask * (1 - fake_tsf_mask)
            # preds = torch.clamp(preds + tsf * front_mask, -1, 1)
            return preds

        def inference(src_inputs, tsf_inputs, T, T_cycle, src_fim, tsf_fim):
            fake_src_color, fake_src_mask, fake_tsf_color, fake_tsf_mask = \
                self.generator.infer_front(src_inputs, tsf_inputs, T=T)

            fake_src_imgs = fake_src_mask * bg_inpaint + (1 - fake_src_mask) * fake_src_color
            fake_tsf_imgs = fake_tsf_mask * bg_inpaint + (1 - fake_tsf_mask) * fake_tsf_color

            if self._opt.front_warp:
                fake_tsf_imgs = warp(fake_tsf_imgs, tsf_inputs[:, 0:3], tsf_fim, fake_tsf_mask)

            cycle_src_inputs, cycle_tsf_inputs = set_cycle_inputs(
                fake_tsf_imgs, src_inputs, tsf_inputs, T_cycle)

            cycle_src_color, cycle_src_mask, cycle_tsf_color, cycle_tsf_mask = \
                self.generator.infer_front(cycle_src_inputs, cycle_tsf_inputs, T=T_cycle)

            cycle_src_imgs = cycle_src_mask * bg_inpaint + (1 - cycle_src_mask) * cycle_src_color
            cycle_tsf_imgs = cycle_tsf_mask * bg_inpaint + (1 - cycle_tsf_mask) * cycle_tsf_color

            if self._opt.front_warp:
                cycle_tsf_imgs = warp(cycle_tsf_imgs, src_inputs[:, 0:3], src_fim, fake_src_mask)

            return fake_src_imgs, fake_tsf_imgs, cycle_src_imgs, cycle_tsf_imgs, fake_src_mask, fake_tsf_mask

        def create_criterion():
            face_criterion = FaceLoss(pretrained_path=self._opt.face_model).cuda()
            idt_criterion = torch.nn.L1Loss()
            mask_criterion = torch.nn.BCELoss()

            return face_criterion, idt_criterion, mask_criterion

        init_lr = 0.0002
        nodecay_epochs = 5
        optimizer = torch.optim.Adam(self.generator.parameters(), lr=init_lr, betas=(0.5, 0.999))
        face_cri, idt_cri, msk_cri = create_criterion()

        step = 0
        logger = tqdm(range(nodecay_epochs))
        for epoch in logger:
            for i, sample in enumerate(data_loader):
                src_fim, tsf_fim, j2ds, T, T_cycle, src_inputs, tsf_inputs, \
                images, init_preds, pseudo_masks = set_gen_inputs(sample)

                # print(bg_inputs.shape, src_inputs.shape, tsf_inputs.shape)
                bs = tsf_inputs.shape[0]
                src_imgs = images[0:bs]
                fake_src_imgs, fake_tsf_imgs, cycle_src_imgs, cycle_tsf_imgs, fake_src_mask, fake_tsf_mask = inference(
                    src_inputs, tsf_inputs, T, T_cycle, src_fim, tsf_fim)

                # cycle reconstruction loss
                cycle_loss = idt_cri(src_imgs, fake_src_imgs) + idt_cri(src_imgs, cycle_tsf_imgs)

                # structure loss
                bg_mask = src_inputs[:, -1:]
                body_mask = 1 - bg_mask
                str_src_imgs = src_imgs * body_mask
                cycle_warp_imgs = F.grid_sample(fake_tsf_imgs, T_cycle)
                back_head_mask = 1 - self.render.encode_front_fim(tsf_fim, transpose=True, front_fn=False)
                struct_loss = idt_cri(init_preds, fake_tsf_imgs) + \
                              2 * idt_cri(str_src_imgs * back_head_mask, cycle_warp_imgs * back_head_mask)

                fid_loss = face_cri(src_imgs, cycle_tsf_imgs, kps1=j2ds[:, 0], kps2=j2ds[:, 0]) + \
                           face_cri(init_preds, fake_tsf_imgs, kps1=j2ds[:, 1], kps2=j2ds[:, 1])

                # mask loss
                # mask_loss = msk_cri(fake_tsf_mask, tsf_inputs[:, -1:]) + msk_cri(fake_src_mask, src_inputs[:, -1:])
                mask_loss = msk_cri(torch.cat([fake_src_mask, fake_tsf_mask], dim=0), pseudo_masks)

                loss = 10 * cycle_loss + 10 * struct_loss + fid_loss + 5 * mask_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if verbose:
                    logger.set_description(
                        (
                            f'epoch: {epoch + 1}; step: {step}; '
                            f'total: {loss.item():.6f}; cyc: {cycle_loss.item():.6f}; '
                            f'str: {struct_loss.item():.6f}; fid: {fid_loss.item():.6f}; '
                            f'msk: {mask_loss.item():.6f}'
                        )
                    )

                if verbose and step % 5 == 0:
                    self.visualize(visualizer, input_imgs=images, tsf_imgs=fake_tsf_imgs, cyc_imgs=cycle_tsf_imgs)

                step += 1

        self.generator.eval()
