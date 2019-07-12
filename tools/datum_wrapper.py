import numpy as np
import pickle


KEY_NAMES = ['pose_keypoints_2d', 'face_keypoints_2d', 'hand_left_keypoints_2d', 'hand_right_keypoints_2d',
             'pose_keypoints_3d', 'face_keypoints_3d', 'hand_left_keypoints_3d', 'hand_right_keypoints_3d',
             'number']


TEMPLATE_POSE25 = np.zeros((1, 25, 3), dtype=np.float32)
TEMPLATE_COCO18 = np.zeros((1, 18, 3), dtype=np.float32)
TEMPLATE_FACE = np.zeros((1, 70, 3), dtype=np.float32)
TEMPLATE_LEFT_RIGHT = np.zeros((1, 21, 3), dtype=np.float32)


def load_pickle_file(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    return data


def write_pickle_file(pkl_path, data_dict):
    with open(pkl_path, 'wb') as fp:
        pickle.dump(data_dict, fp, protocol=2)


def get_default_info():

    datum_info = dict()
    for key in KEY_NAMES:
        datum_info[key] = []

    return datum_info


def write_datum_sample(datum_info, datum, only_body=False, is_coco=False, is_first=False):
    # print("Body keypoints: \n" + str(datum.poseKeypoints.shape))
    # print("Face keypoints: \n" + str(datum.faceKeypoints.shape))
    # print("Left hand keypoints: \n" + str(datum.handKeypoints[0].shape))
    # print("Right hand keypoints: \n" + str(datum.handKeypoints[1].shape))

    number = 1

    poseKps = datum.poseKeypoints
    faceKps = datum.faceKeypoints
    leftHandKps = datum.handKeypoints[0]
    rightHandKps = datum.handKeypoints[1]

    if len(poseKps.shape) == 0:
        if is_first:
            if only_body:
                poseKps = TEMPLATE_COCO18 if is_coco else TEMPLATE_POSE25
            else:
                poseKps = TEMPLATE_COCO18 if is_coco else TEMPLATE_POSE25
                faceKps = TEMPLATE_FACE
                leftHandKps = TEMPLATE_LEFT_RIGHT
                rightHandKps = TEMPLATE_LEFT_RIGHT
        else:
            if only_body:
                poseKps = datum_info['pose_keypoints_2d'][-1]
            else:
                poseKps = datum_info['pose_keypoints_2d'][-1]
                faceKps = datum_info['face_keypoints_2d'][-1]
                leftHandKps = datum_info['hand_left_keypoints_2d'][-1]
                rightHandKps = datum_info['hand_right_keypoints_2d'][-1]

        number = 0
        print('the number of people is 0 ...')

    elif poseKps.shape[0] > 1:
        if only_body:
            poseKps = poseKps[0:1]
        else:
            poseKps = poseKps[0:1]
            faceKps = faceKps[0:1]
            leftHandKps = leftHandKps[0:1]
            rightHandKps = rightHandKps[0:1]

        number = poseKps.shape[0]
        print('the number of people {} is more than 1 ...'.format(number))

    if only_body:
        datum_info['pose_keypoints_2d'].append(poseKps)
    else:
        datum_info['pose_keypoints_2d'].append(poseKps)
        datum_info['face_keypoints_2d'].append(faceKps)
        datum_info['hand_left_keypoints_2d'].append(leftHandKps)
        datum_info['hand_right_keypoints_2d'].append(rightHandKps)

    datum_info['number'].append(number)

    return number


def write_result_info_sample(datum_info, result_info, only_body=False, is_coco=False, is_first=False):
    # print("Body keypoints: \n" + str(datum.poseKeypoints.shape))
    # print("Face keypoints: \n" + str(datum.faceKeypoints.shape))
    # print("Left hand keypoints: \n" + str(datum.handKeypoints[0].shape))
    # print("Right hand keypoints: \n" + str(datum.handKeypoints[1].shape))

    number = 1

    poseKps = result_info['pose_keypoints_2d']
    faceKps = result_info['face_keypoints_2d']
    leftHandKps = result_info['hand_left_keypoints_2d']
    rightHandKps = result_info['hand_right_keypoints_2d']

    if len(poseKps.shape) == 0:
        if is_first:
            if only_body:
                poseKps = TEMPLATE_COCO18 if is_coco else TEMPLATE_POSE25
            else:
                poseKps = TEMPLATE_COCO18 if is_coco else TEMPLATE_POSE25
                faceKps = TEMPLATE_FACE
                leftHandKps = TEMPLATE_LEFT_RIGHT
                rightHandKps = TEMPLATE_LEFT_RIGHT
        else:
            if only_body:
                poseKps = datum_info['pose_keypoints_2d'][-1]
            else:
                poseKps = datum_info['pose_keypoints_2d'][-1]
                faceKps = datum_info['face_keypoints_2d'][-1]
                leftHandKps = datum_info['hand_left_keypoints_2d'][-1]
                rightHandKps = datum_info['hand_right_keypoints_2d'][-1]

        number = 0
        print('the number of people is 0 ...')

    elif poseKps.shape[0] > 1:
        if only_body:
            poseKps = poseKps[0:1]
        else:
            poseKps = poseKps[0:1]
            faceKps = faceKps[0:1]
            leftHandKps = leftHandKps[0:1]
            rightHandKps = rightHandKps[0:1]

        number = poseKps.shape[0]
        print('the number of people {} is more than 1 ...'.format(number))

    if only_body:
        datum_info['pose_keypoints_2d'].append(poseKps)
    else:
        datum_info['pose_keypoints_2d'].append(poseKps)
        datum_info['face_keypoints_2d'].append(faceKps)
        datum_info['hand_left_keypoints_2d'].append(leftHandKps)
        datum_info['hand_right_keypoints_2d'].append(rightHandKps)

    datum_info['number'].append(number)

    return number


def write_datum_pkl(pkl_path, datum_info):
    for i in range(4):
        key = KEY_NAMES[i]
        if len(datum_info[key]) > 0:
            if key == 'number':
                datum_info[key] = np.array(datum_info[key])
            else:
                datum_info[key] = np.concatenate(datum_info[key], axis=0)
            print('\t', key, datum_info[key].shape)

    write_pickle_file(pkl_path, datum_info)
    print('write video outputs to {}'.format(pkl_path))


def write_add_number_pkl(pkl_path):
    datum_info = load_pickle_file(pkl_path)

    if 'number' not in datum_info:
        key = KEY_NAMES[0]
        length = datum_info[key].shape[0]
        numbers = np.ones(length, dtype=np.float32)

        datum_info['number'] = numbers

        write_pickle_file(pkl_path, datum_info)
        print('write video outputs to {}'.format(pkl_path))


def write_fail_case(file_path, fail_case_list):
    with open(file_path, 'w') as writer:
        for image_path, number in fail_case_list:
            line = '%s    %d\n' % (image_path, number)
            writer.write(line)
            print(line[:-1])
