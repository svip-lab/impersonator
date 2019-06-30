import cv2
import numpy as np
import os
import glob


SELECTED_VIDEO_NAMES = ['023_3_1', '024_8_2', '026_1_1']

def get_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


def images_to_video(images_paths, out_path, image_size=256):
    fps = 24
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    """create video"""
    videoWriter = cv2.VideoWriter(out_path, fourcc, fps, (image_size, image_size))

    length = len(images_paths)
    for i, image_path in enumerate(images_paths):
        frame = cv2.imread(image_path)

        videoWriter.write(frame)
        print(i, length)

    print('save video to {}'.format(out_path))


def parse_images_paths(image_dir, key_words='pred'):
    images_paths = []
    for i, image_name in enumerate(os.listdir(image_dir)):
        if key_words in image_name:
            image_path = os.path.join(image_dir, image_name)
            images_paths.append(image_path)

    images_paths.sort()
    return images_paths


def convert(root_dir, out_dir, model, demo_name, is_gt=False):
    if is_gt:
        key = 'gt'
    else:
        key = 'pred'

    # global SELECTED_VIDEO_NAMES
    images_dirs = os.path.join(root_dir, model, demo_name)
    video_dirs = get_dir(os.path.join(out_dir, model, demo_name))
    for video_name in SELECTED_VIDEO_NAMES:
        images_paths_dir = os.path.join(images_dirs, video_name)
        images_paths = parse_images_paths(images_paths_dir, key_words=key)

        video_path = os.path.join(video_dirs, video_name + '.avi')
        images_to_video(images_paths, video_path)
        print(video_path)


def convert_gt(root_dir, out_dir):
    images_paths = parse_images_paths(root_dir, key_words='gt')
    images_to_video(images_paths, out_dir)
    print(out_dir)


if __name__ == '__main__':
    root_dir = 'D:\\OneDrive\\projects\\experiments\\impersonator\\dance_demo_outs'
    out_dir = 'D:\\OneDrive\\projects\\experiments\\impersonator\\dance_demo_outs_videos'

    # for model in ['pG2', 'SHUP', 'DSC', 'OURS']:
    #     for demo_name in ['dance_demo_6', 'dance_demo_7']:
    #         convert(root_dir, out_dir, model, demo_name, is_gt=False)

    for model in ['OURS_2']:
        for demo_name in ['dance_demo_6']:
            convert(root_dir, out_dir, model, demo_name, is_gt=False)

    # for demo_name in ['dance_demo_6', 'dance_demo_7']:
    #     images_paths = parse_images_paths(os.path.join(root_dir, 'DSC', demo_name, '024_8_2'), key_words='gt')
    #     images_to_video(images_paths, os.path.join(out_dir, demo_name + '.avi'))