import numpy as np
import torch
import os

from networks.bodymesh.hmr import HumanModelRecovery
from utils.util import load_pickle_file, write_pickle_file


os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '3'


hmr = HumanModelRecovery('pretrains/smpl_model.pkl', feature_dim=2048, theta_dim=85, iterations=3).cuda()
saved_data = torch.load('pretrains/hmr_tf2pt.pth')
hmr.load_state_dict(saved_data)
hmr.eval()

data_dir = '/p300/human_pose/processed'
smpl_dir = os.path.join(data_dir, 'motion_transfer_smpl')


def cal_kps(smpl_data, batch_size=1024):
    # ['pose', 'shape', 'cams', 'vertices']
    pose = smpl_data['pose']
    shape = smpl_data['shape']
    cams = smpl_data['cams']

    thetas = torch.FloatTensor(np.concatenate([cams, pose, shape], axis=1)).cuda()
    length = thetas.shape[0]

    kps = []
    num_iters = int(np.ceil(length / batch_size))

    for i in range(num_iters):
        batch_thetas = thetas[i*batch_size: (i+1)*batch_size]

        out_info = hmr.get_details(batch_thetas)
        kps.append(out_info['j2d'])

    kps = torch.cat(kps, dim=0)
    return kps


def converts():

    for p_id in os.listdir(smpl_dir):
        smpl_p_path = os.path.join(smpl_dir, p_id)

        for c_id in os.listdir(smpl_p_path):
            smpl_c_path = os.path.join(smpl_p_path, c_id)

            for a_id in os.listdir(smpl_c_path):
                smpl_a_path = os.path.join(smpl_c_path, a_id)

                smpl_file = os.path.join(smpl_a_path, 'pose_shape.pkl')
                kps_file = os.path.join(smpl_a_path, 'kps.pkl')

                smpl_data = load_pickle_file(smpl_file)

                kps = cal_kps(smpl_data, batch_size=4096)
                kps = kps.cpu().numpy()
                write_pickle_file(kps_file, {'kps': kps})
                print(smpl_file, kps_file, kps.shape)



converts()