from concurrent import futures
from math import ceil
import cv2
import sys
import os
import argparse
import time
import progressbar
import numpy as np

# /p300/action_transfer_dataset/1080x1080/1/1.mp4

"""
python tools/convert_video2image.py \
    --source=/p300/action_transfer_dataset/1080x1080 \
    --dest_folder=/p300/human_pose/processed/motion_transfer_HD \
    --image_size=-300

"""


def parser_args():
    '''

    python tools/convert_video2image.py \
        --source=/p300/action_transfer_dataset/1080x1080 \
        --dest_folder=/p300/human_pose/processed/motion_transfer_HD \
        --image_size=1024

    python tools/convert_video2image.py \
    --source=/home/liuwen/ssd/human_pose/Dancing/human/videos/ \
    --dest_folder=/home/liuwen/ssd/human_pose/Dancing/human/frames/

    :return:
    '''
    parser = argparse.ArgumentParser(description='convert video to image.')
    parser.add_argument('--source_folder', help='the path of video folder.')
    parser.add_argument('--dest_folder', help='the  path of destination folder, which saves the images converted by video.')
    parser.add_argument('--image_size', type=int, default=0, help='resize image or not, if is 0, do not resize image')

    return parser.parse_args()


def validate_video(video_name):
    capture = cv2.VideoCapture()
    if capture.isOpened():
        capture.release()
    capture.open(video_name)

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    return total_frames != 0


def validate_video_folder(video_folder, dest_folder):
    assert os.path.exists(video_folder), 'video folder {} does not exist.'.format(video_folder)
    if not os.path.exists(dest_folder):
        os.mkdir(dest_folder)

    valid_video_list = []
    log_file = 'invalid video name:\n'
    invalid = False

    user_list = os.listdir(video_folder)
    for user_name in user_list:
        print('validating on user = {}'.format(user_name))

        # user_dir = video_folder + user_name
        user_dir = os.path.join(video_folder, user_name)

        for pose in os.listdir(user_dir):

            pose_dir = os.path.join(user_dir, pose)

            for filename in os.listdir(pose_dir):
                video_name = os.path.join(video_folder, user_name, pose, filename)
                if validate_video(video_name):
                    dest_video_folder = os.path.join(dest_folder, user_name, pose, filename.split('.')[0])
                    if not os.path.exists(dest_video_folder):
                        os.makedirs(dest_video_folder)

                    valid_video_list.append((video_name, dest_video_folder))

                    print('validate {} is ok.'.format(video_name))
                else:
                    invalid = True
                    log_file += video_name + '\n'
                    response = input("'{}' encounters to some problems, can not convert to images, \n"
                                     "are you going to continue? y(or n).".format(video_name))

                    if response == 'n':
                        sys.exit(-1)

    if invalid:
        with open('{}/log.txt'.format(dest_folder), 'w') as f:
            f.write(log_file)

    return valid_video_list


def write_video_image(video_save_tuple, image_size=0):
    video_name = video_save_tuple[0]
    save_folder = video_save_tuple[1]

    print('processing {}, and saving to {}'.format(video_name, save_folder))

    capture = cv2.VideoCapture()
    if capture.isOpened():
        capture.release()
    capture.open(video_name)

    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    cur_frame = 0
    index_frame = 0
    total_frames_digits = len(str(total_frames))

    # progress bar
    bar = progressbar.ProgressBar(maxval=total_frames,
                                  widgets=[progressbar.Bar('>', '[', ']'), ' ', progressbar.SimpleProgress(), ' ',
                                           progressbar.Percentage(), ' ', progressbar.ETA()]).start()
    while cur_frame < total_frames:
        bar.update(cur_frame)
        retval, frame = capture.read()
        if retval:
            image_name = '{}/{:0>%d}.jpg' % total_frames_digits
            image_name = image_name.format(save_folder, index_frame)

            if video_name.endswith('.MOV'):
                frame = np.transpose(frame, axes=[1, 0, 2])

            if image_size > 0:
                frame = cv2.resize(frame, (image_size, image_size))

            cv2.imwrite(image_name, frame)
            index_frame += 1
        cur_frame += 1

    bar.finish()
    capture.release()
    print('saving {} into {}, total frames is {}.'.format(video_name, save_folder, index_frame))


def convert_video_image(video_folder, dest_folder, image_size):
    valid_video_list = validate_video_folder(video_folder, dest_folder)

    total_video = len(valid_video_list)
    # cpu_numbers = futures.process.multiprocessing.cpu_count()
    cpu_numbers = 103
    pool_folder = int(ceil(total_video / cpu_numbers))

    print('converting {} videos into images by {} process.'.format(total_video, cpu_numbers))

    start_time = time.time()
    for idx in range(pool_folder):
        begin_idx = idx * cpu_numbers
        end_idx = (idx + 1) * cpu_numbers
        if end_idx >= total_video:
            end_idx = total_video

        print('computing {} pool folder, {}: {}'.format(idx, begin_idx, end_idx))
        with futures.ProcessPoolExecutor() as pool:
            pool.map(write_video_image, valid_video_list[begin_idx: end_idx], [image_size] * (end_idx - begin_idx))

        # for video, image_size in zip(valid_video_list[begin_idx: end_idx], [image_size] * (end_idx - begin_idx)):
        #     write_video_image(video, image_size)

    print('using parallel to computing, using time = {}...'.format(time.time() - start_time))


if __name__ == '__main__':
    args = parser_args()
    source_folder = args.source_folder
    dest_folder = args.dest_folder
    image_size = args.image_size

    print(source_folder)
    print(dest_folder)
    convert_video_image(source_folder, dest_folder, image_size)
