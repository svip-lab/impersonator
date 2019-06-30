import os
import cv2


data_dir = '/p300/human_pose/processed'
md_dir = os.path.join(data_dir, 'motion_transfer')
mdhd_dir = os.path.join(data_dir, 'motion_transfer_HD')

count = 0


def check():
    global count

    for p_id in os.listdir(md_dir):
        md_p_path = os.path.join(md_dir, p_id)
        mdhd_p_path = os.path.join(mdhd_dir, p_id)

        assert os.path.join(mdhd_p_path), '{} dose not exist!'.format(mdhd_p_path)

        for c_id in os.listdir(md_p_path):
            md_c_path = os.path.join(md_p_path, c_id)
            mdhd_c_path = os.path.join(mdhd_p_path, c_id)

            assert os.path.join(mdhd_c_path), '{} dose not exist!'.format(mdhd_c_path)

            for a_id in os.listdir(md_c_path):
                md_a_path = os.path.join(md_c_path, a_id)
                mdhd_a_path = os.path.join(mdhd_c_path, a_id)

                assert os.path.join(mdhd_a_path), '{} dose not exist!'.format(mdhd_a_path)

                md_images = sorted(os.listdir(md_a_path))
                mdhd_images = sorted(os.listdir(mdhd_a_path))

                print(p_id, c_id, a_id, len(md_images), len(mdhd_images))

                count += len(md_images)

    print('total = {}'.format(count))


check()