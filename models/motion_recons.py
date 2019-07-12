import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict

import utils.util as util
import utils.cv_utils as cv_utils
from utils.pose_utils import PoseEstimator
from models.models import BaseRunnerModel, BaseModel, BaseTrainerModel
import networks
import networks.losses as losses

import ipdb


class MotionReconsSolverUnit(nn.Module):
    def __init__(self, init_states, init_smpls, init_rs, init_j3d):
        super(MotionReconsSolverUnit, self).__init__()

        # set states as parameters and initialized by init_states
        self.states = nn.Parameter(init_states)

        self.register_buffer('init_states', init_states)
        self.register_buffer('init_smpls', init_smpls)
        self.init_rs = init_rs
        self.init_j3d = init_j3d

    @staticmethod
    def compute_2d_loss(pred_j2d, gt_j2d):
        """
        :param pred_j2d: torch.FloatTensor, (N, number of points, 2), 2=xy
        :param gt_j2d: torch.FloatTensor, (N, number of points, 3), 3=xyc
        :return:
        """
        return gt_j2d[:, :, 2:3] * torch.abs(pred_j2d[:, :, 0:2] - gt_j2d[:, :, 0:2])

    @staticmethod
    def compute_3d_loss(pred_rots, init_rots, cof):
        """
        :param pred_rots:
        :param init_rots:
        :param cof:
        :return:
        """
        return cof * torch.abs(pred_rots - init_rots)

    @staticmethod
    def compute_sm_loss(pred_j3d, init_j3d):
        """
        :param pred_j3d:
        :param init_j3d:
        :return:
        """
        return (pred_j3d[0:-1] - init_j3d[1:]) ** 2

    def forward(self, hmr, observed_poses):
        # 1. get the current smpl and skin blended outs
        cur_smpl = hmr.regressor(self.states)
        skin_blend_outs = hmr.skin_blend(cur_smpl, get_rs=True)

        # 2. calculate the losses of each part
        ### 2.1 2D loss
        l2d = self.compute_2d_loss(skin_blend_outs['j2d'], observed_poses)

        ### 2.2 3D loss
        l3d = self.compute_3d_loss(skin_blend_outs['rs'], self.init_rs, l2d)

        ### 2.3 smooth loss
        lsm = self.compute_sm_loss(skin_blend_outs['j3d'], self.init_j3d)

        return cur_smpl, torch.mean(l2d), torch.mean(l3d), torch.mean(lsm)


class MotionReconsSolver(object):

    def __init__(self, hmr,
                 lr=0.0001,
                 w2d=10.0, w3d=100.0, wsm=25.0,
                 num_iters=1000):

        self.hmr = hmr
        self.lr = lr
        self.w2d = w2d
        self.w3d = w3d
        self.wsm = wsm
        self.num_iters = num_iters

    def solve(self, init_details, observed_poses):
        # 1. define motion recons unit
        init_states = init_details['states']
        init_smpls = init_details['theta']
        init_rs = init_details['rs']
        init_j3d = init_details['j3d']
        mr_unit = MotionReconsSolverUnit(init_states, init_smpls, init_rs=init_rs, init_j3d=init_j3d)

        # 2. define optimizer
        optimizer = torch.optim.Adam(mr_unit.parameters(), lr=self.lr)

        # 3. iteration-loop for optimization
        cur_smpl = init_smpls
        for i in range(self.num_iters):
            # 3.1 run motion reconstruction unit
            cur_smpl, l2d, l3d, lsm = mr_unit(self.hmr, observed_poses)

            # 3.2 compute the total loss
            total_loss = self.w2d * l2d + self.w3d * l3d + self.wsm * lsm

            # 3.3 optimizer step
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            print(i, total_loss)

        return cur_smpl


class MotionRecons(object):
    def __init__(self, opt):

        self._opt = opt
        self.hmr = self._create_hmr()
        self.pose_estimator = self._create_pose_estimator()

    def _create_hmr(self):
        hmr = networks.HumanModelRecovery(smpl_data=util.load_pickle_file(self._opt.smpl_model))
        saved_data = torch.load(self._opt.hmr_model)
        hmr.load_state_dict(saved_data)
        hmr.eval()
        return hmr

    def _create_pose_estimator(self):
        pose_estimator = PoseEstimator()
        return pose_estimator

    # TODO: need to be implemented
    def preprocessing(self, orginal_images, pose_puts):
        """
        image preprocessing, including image crop by estimated pose, and resize to 224 x 224

        :param orginal_images:
        :param pose_puts:
        :return:
        """
        images = []
        poses = []
        return images, poses

    # TODO: need to be implemented
    def load_images(self, images_paths):
        images = []
        return images

    def forward(self, *input):
        pass

    def optimize(self, images_paths, pose_outs=None):
        # 1. pose estimator
        if pose_outs is None:
            pose_outs = self.pose_estimator.estimate_multiples(images_paths)

        # 2. load images preprocessing
        original_images = self.load_images(images_paths)
        images, poses = self.preprocessing(original_images, pose_outs)

        # 3. estimate the initial hmr hidden state and initial SMPL out
        images = torch.FloatTensor(images).cuda()
        poses = torch.FloatTensor(poses).cuda()
        init_states = self.hmr.encode(images)
        init_smpls = self.hmr.regressor(init_states)
        init_details = self.hmr.get_details(init_smpls, get_rs=True)
        init_details['states'] = init_states

        # 4. solver
        solver = MotionReconsSolver(self.hmr, num_iters=5000)
        final_smpl = solver.solve(init_details, poses)

        return final_smpl


class PoseReconsModel(torch.nn.Module):

    def __init__(self, opt):
        super(PoseReconsModel, self).__init__()
        self._name = 'PoseReconsModel'
        self._opt = opt

        # create networks
        self._init_create_networks()

    def _create_hmr(self):
        hmr = networks.HumanModelRecovery(smpl_data=util.load_pickle_file(self._opt.smpl_model))
        saved_data = torch.load(self._opt.hmr_model)
        hmr.load_state_dict(saved_data)
        hmr.eval()
        return hmr

    def _create_render(self, faces):
        render = networks.SMPLRendererTrainer(faces=faces, map_name=self._opt.map_name,
                                              uv_map_path=self._opt.uv_mapping, tex_size=self._opt.tex_size,
                                              image_size=self._opt.image_size, fill_back=True,
                                              anti_aliasing=True, background_color=(0, 0, 0),
                                              has_front_map=self._opt.wrap_face)

        return render

    def _init_create_networks(self):
        # hmr and render
        self.hmr = self._create_hmr()
        self.render = self._create_render(self.hmr.smpl.faces)

    def _create_optimizer(self):
        self._current_lr_G = self._opt.lr_G

        # initialize optimizers
        self._optimizer = torch.optim.Adam(self._G.parameters(), lr=self._current_lr_G,
                                           betas=[self._opt.G_adam_b1, self._opt.G_adam_b2])

    def optimize(self):
        pass
