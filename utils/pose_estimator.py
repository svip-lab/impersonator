import cv2
import numpy as np

from openpose import pyopenpose as op


MODEL_POSE = ['BODY_25', 'COCO', 'MPI']
OPENPOSE_RESULT_KEYS = ['pose_keypoints_2d', 'face_keypoints_2d', 'hand_left_keypoints_2d', 'hand_right_keypoints_2d',
                        'pose_keypoints_3d', 'face_keypoints_3d', 'hand_left_keypoints_3d', 'hand_right_keypoints_3d']

RESULT_KEYS = ['pose_keypoints_2d', 'face_keypoints_2d', 'hand_left_keypoints_2d', 'hand_right_keypoints_2d']
KEY_MAPPINGS = {
    'pose': 'pose_keypoints_2d',
    'hand': 'face_keypoints_2d',
    'left_hand': 'hand_left_keypoints_2d',
    'right_hand': 'hand_right_keypoints_2d'
}

# parameters
params = dict()
params['model_folder'] = '/public/liuwen/openpose/models'
params['model_pose'] = 'BODY_25'
params['face'] = True
params['hand'] = True
params['keypoint_scale'] = 4

# wrapper
opWrapper = op.WrapperPython()

# datum
datum = op.Datum()

# check is start
is_run = False


def config(new_params=None):
    global params, opWrapper, is_run
    if new_params is not None:
        for key, value in new_params.items():
            params[key] = value

    if is_run:
        close()
    else:
        is_run = True

    opWrapper.configure(params)
    opWrapper.start()


def close():
    global opWrapper, is_run
    opWrapper.stop()
    is_run = False


def estimate(image_path):
    """
    :param image_path:
    :return:
    """
    global opWrapper, datum

    imageToProcess = cv2.imread(image_path)
    datum.cvInputData = imageToProcess
    opWrapper.emplaceAndPop([datum])

    result_info = {
        'pose_keypoints_2d': datum.poseKeypoints,
    }

    if params['face']:
        result_info['face_keypoints_2d'] = datum.faceKeypoints

    if params['hand']:
        result_info['hand_left_keypoints_2d'] = datum.handKeypoints[0]
        result_info['hand_right_keypoints_2d'] = datum.handKeypoints[1]

    return result_info


def estimate_multiples(images_paths):
    """
    :param images_paths: list, list of str, the list of image paths
    :return:
    """
    global opWrapper, datum

    # 'pose_keypoints_2d', 'face_keypoints_2d', 'hand_left_keypoints_2d', 'hand_right_keypoints_2d',
    result_info = {
        'pose_keypoints_2d': [],
        'face_keypoints_2d': [],
        'hand_left_keypoints_2d': [],
        'hand_right_keypoints_2d': []
    }
    for image_path in images_paths:
        single_info = estimate(image_path)
        result_info['pose_keypoints_2d'].append(single_info['pose_keypoints_2d'])

        if params['face']:
            result_info['face_keypoints_2d'].append(single_info['face_keypoints_2d'])

        if params['hand']:
            result_info['hand_left_keypoints_2d'].append(single_info['hand_left_keypoints_2d'])
            result_info['hand_right_keypoints_2d'].append(single_info['hand_right_keypoints_2d'])

    result_info['pose_keypoints_2d'] = np.concatenate(result_info['pose_keypoints_2d'])

    if params['face']:
        result_info['face_keypoints_2d'] = np.concatenate(result_info['face_keypoints_2d'])

    if params['hand']:
        result_info['hand_left_keypoints_2d'] = np.concatenate(result_info['hand_left_keypoints_2d'])
        result_info['hand_right_keypoints_2d'] = np.concatenate(result_info['hand_right_keypoints_2d'])

    return result_info


if __name__ == '__main__':
    import os
    os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = '8'

    src_img_path = '/public/liuwen/p300/human_pose/processed/motion_transfer_HD/001/1/1/0000.jpg'

    config()

    for i in range(10):
        results = estimate(src_img_path)
        print(i, src_img_path, results['pose_keypoints_2d'].shape)


