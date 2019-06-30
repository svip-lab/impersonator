import torch
import torch.nn as nn
from networks.bodymesh import HumanModelRecovery


class HMRLoss(nn.Module):
    def __init__(self, pretrain_model, smpl_data):
        super(HMRLoss, self).__init__()
        self.hmr = HumanModelRecovery(smpl_data=smpl_data)
        self.load_model(pretrain_model)
        self.criterion = nn.L1Loss()
        self.eval()

    def forward(self, x, y):
        x_hmr, y_hmr = self.hmr(x), self.hmr(y)
        loss = 0.0
        for i in range(len(x_hmr)):
            loss += self.criterion(x_hmr[i], y_hmr[i].detach())
            # loss += self.criterion(x_hmr[i], y_hmr[i])
        return loss

    def load_model(self, pretrain_model):
        saved_data = torch.load(pretrain_model)
        self.hmr.load_state_dict(saved_data)
        print('load hmr model from {}'.format(pretrain_model))