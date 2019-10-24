import cv2
import torch
import numpy as np
import os
import os.path as osp
import glob
from tqdm import tqdm
import time
from utils.visdom_visualizer import VisdomVisualizer
from utils.nmr import SMPLRenderer
from networks.hmr import HumanModelRecovery

os.environ['CUDA_DEVICES_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


IMG_SIZE = 256
visualizer = VisdomVisualizer(env='visual', ip='http://10.10.10.100', port=31102)


def visual(model, out_dir):
    global visualizer

    render = SMPLRenderer(image_size=IMG_SIZE).cuda()

    texs = render.debug_textures().cuda()[None]

    with h5py.File(osp.join(out_dir, 'smpl_infos.h5'), 'r') as reader:
        cams_crop = reader['cam_crop']
        poses = reader['pose']
        shapes = reader['shape']
        frame_ids = reader['f_id']

        scan_image_paths = sorted(glob.glob(osp.join(out_dir, 'cropped_frames', '*.png')))

        for i in range(len(frame_ids) - 1):
            assert frame_ids[i] < frame_ids[i + 1]

        image_paths = [scan_image_paths[f_id] for f_id in frame_ids]

        for i in tqdm(range(len(image_paths))):
            im_path = image_paths[i]
            image = cv2.imread(im_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = np.transpose(image, (2, 0, 1))
            image = image.astype(np.float) / 255
            image = torch.tensor(image).float()[None].cuda()

            cams = torch.tensor(cams_crop[i]).float()[None].cuda()
            pose = torch.tensor(poses[i]).float()[None].cuda()
            shape = torch.tensor(shapes[i]).float()[None].cuda()
            verts, _, _ = model.smpl(beta=shape, theta=pose, get_skin=True)
            rd_imgs, _ = render.render(cams, verts, texs.clone())
            sil = render.render_silhouettes(cams, verts)

            masked_img = image * sil[:, None, :, :]

            visualizer.vis_named_img('rd_imgs', rd_imgs, denormalize=False)
            visualizer.vis_named_img('masked_img', masked_img, denormalize=False)
            visualizer.vis_named_img('imgs', image, denormalize=False)

            time.sleep(1)


if __name__ == '__main__':

    hmr = HumanModelRecovery(smpl_pkl_path='./assets/pretrains/smpl_model.pkl')
    hmr.load_state_dict(torch.load('./assets/pretrains/hmr_tf2pt.pth'))
    hmr = hmr.eval().cuda()

    ## process neuralAvatar
    src_path = '/p300/tpami/neuralAvatar/original/wenliu_fps_30.mp4'
    out_dir = '/p300/tpami/neuralAvatar/processed/wenliu_fps_30'

    # src_path = '/p300/tpami/neuralAvatar/processed/wenliu_fps_30/frames/frame00000000.png'
    # out_dir = '/p300/tpami/neuralAvatar/processed/frame00000000'
    rescale = None
    is_visual = True

    process(hmr,
            src_path=src_path,
            output_dir=out_dir,
            save_crop=True,
            rescale=rescale)

    if is_visual:
        visual(hmr, out_dir=out_dir)
