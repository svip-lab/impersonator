import os
import glob
from tqdm import tqdm


# Replacing them as your own folder
dataset_video_root_path = '/p300/tpami/iPER_examples/iPER_256_video_release'
save_images_root_path = '/p300/tpami/iPER_examples/images'


def extract_one_video(video_path, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    os.system("ffmpeg -i %s -start_number 0 %s/frame%%08d.png > /dev/null 2>&1" % (video_path, save_dir))


def main():
    global dataset_video_root_path, save_images_root_path

    video_path_list = sorted(glob.glob("%s/*.mp4" % dataset_video_root_path))

    for video_path in tqdm(video_path_list):
        video_name = os.path.split(video_path)[-1][:-4]
        actor_id, cloth_id, action_type = video_name.split('_')

        video_images_dir = os.path.join(save_images_root_path, actor_id, cloth_id, action_type)
        extract_one_video(video_path, video_images_dir)

        # import ipdb
        # ipdb.set_trace()


if __name__ == '__main__':
    main()

