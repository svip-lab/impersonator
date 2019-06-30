import cv2
import numpy as np

from openpose import pyopenpose as op


class PoseEstimator(object):
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

    def __init__(self, model_pose='BODY_25', face=False, hand=False, keypoint_scale=4):
        self.is_run = False
        self.opWrapper = op.WrapperPython()
        self.params = dict()
        # model configure
        self.params['model_folder'] = '/public/liuwen/openpose/models'
        self.params['model_pose'] = model_pose
        self.params['face'] = face
        self.params['hand'] = hand

        # output configure
        self.params['keypoint_scale'] = keypoint_scale

        # configure and start params
        self.config()

        # Process Image
        self.datum = op.Datum()

    def config(self, params=None):
        if params is not None:
            for key, value in params.items():
                self.params[key] = value

        if self.is_run:
            self.close()
        else:
            self.is_run = True

        self.opWrapper.configure(self.params)
        self.opWrapper.start()

    def close(self):
        self.opWrapper.stop()
        self.is_run = False

    def run(self, image_path):
        """
        :param image_path:
        :return:
        """
        imageToProcess = cv2.imread(image_path)
        self.datum.cvInputData = imageToProcess
        self.opWrapper.emplaceAndPop([self.datum])
        return self.datum

    def estimate(self, image_path):
        """
        :param image_path:
        :return:
        """
        imageToProcess = cv2.imread(image_path)
        self.datum.cvInputData = imageToProcess
        self.opWrapper.emplaceAndPop([self.datum])

        result_info = {
            'pose_keypoints_2d': self.datum.poseKeypoints,
        }

        if self.params['face']:
            result_info['face_keypoints_2d'] = self.datum.faceKeypoints

        if self.params['hand']:
            result_info['hand_left_keypoints_2d'] = self.datum.handKeypoints[0]
            result_info['hand_right_keypoints_2d'] = self.datum.handKeypoints[1]

        return result_info

    def estimate_multiples(self, images_paths):
        """
        :param images_paths: list, list of str, the list of image paths
        :return:
        """
        # 'pose_keypoints_2d', 'face_keypoints_2d', 'hand_left_keypoints_2d', 'hand_right_keypoints_2d',
        result_info = {
            'pose_keypoints_2d': [],
            'face_keypoints_2d': [],
            'hand_left_keypoints_2d': [],
            'hand_right_keypoints_2d': []
        }
        for image_path in images_paths:
            single_info = self.estimate(image_path)
            result_info['pose_keypoints_2d'].append(single_info['pose_keypoints_2d'])

            if self.params['face']:
                result_info['face_keypoints_2d'].append(single_info['face_keypoints_2d'])

            if self.params['hand']:
                result_info['hand_left_keypoints_2d'].append(single_info['hand_left_keypoints_2d'])
                result_info['hand_right_keypoints_2d'].append(single_info['hand_right_keypoints_2d'])

        result_info['pose_keypoints_2d'] = np.concatenate(result_info['pose_keypoints_2d'])

        if self.params['face']:
            result_info['face_keypoints_2d'] = np.concatenate(result_info['face_keypoints_2d'])

        if self.params['hand']:
            result_info['hand_left_keypoints_2d'] = np.concatenate(result_info['hand_left_keypoints_2d'])
            result_info['hand_right_keypoints_2d'] = np.concatenate(result_info['hand_right_keypoints_2d'])

        return result_info


if __name__ == '__main__':
    import os
    os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = '8'

    # pose_estimator = PoseEstimator()
    pose_estimator = PoseEstimator(model_pose='BODY_25', face=True, hand=True, keypoint_scale=4)

    src_img_path = '/public/liuwen/p300/human_pose/processed/motion_transfer_HD/001/1/1/0000.jpg'

    for i in range(10):
        results = pose_estimator.estimate(src_img_path)
        print(i, src_img_path, results['pose_keypoints_2d'].shape)


