import os
import glob
from tqdm import tqdm
import subprocess

import shutil


# Replacing them as your own folder
dataset_video_root_path = '/p300/tpami/iPER_ICCV_TEST/iPER_256_video_release'
save_images_root_path = '/p300/tpami/iPER_ICCV_TEST/images'


def extract_one_video(video_path, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    # os.system('ffmpeg -i %s -start_number 0 %s/frame%%08d.png > /dev/null 2>&1' % (video_path, save_dir))

    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-start_number", "0",
        "{save_dir}/frame_%08d.png".format(save_dir=save_dir),
        "-loglevel", "quiet"
    ]

    print(" ".join(cmd))
    subprocess.run(cmd)

    # rename it to comparable with Protocol his_evaluators
    images_names = os.listdir(save_dir)
    images_names.sort()

    num_images = len(images_names)
    num_digits = len(str(num_images))
    image_name_template = '{:0>%d}.jpg' % num_digits

    for i, img_name in enumerate(images_names):
        src_img_path = os.path.join(save_dir, img_name)
        tgt_img_path = os.path.join(save_dir, image_name_template.format(i))

        shutil.move(src_img_path, tgt_img_path)
        # print(src_img_path, tgt_img_path)


def main():
    global dataset_video_root_path, save_images_root_path

    video_path_list = sorted(glob.glob('%s/*.mp4' % dataset_video_root_path))

    for video_path in tqdm(video_path_list):
        video_name = os.path.split(video_path)[-1][:-4]
        actor_id, cloth_id, action_type = video_name.split('_')

        video_images_dir = os.path.join(save_images_root_path, actor_id, cloth_id, action_type)
        extract_one_video(video_path, video_images_dir)

        # import ipdb
        # ipdb.set_trace()


if __name__ == '__main__':
    main()

