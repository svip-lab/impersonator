import os
import shutil


pG2_dir = '/home/piaozx/liuwen/p300/results/pG2'
ours_dir = '/home/piaozx/liuwen/p300/results/impersonator_02_29'


for video_name in os.listdir(pG2_dir):

    pG2_vid_path = os.path.join(pG2_dir, video_name)
    ours_vid_path = os.path.join(ours_dir, video_name)

    # copy, 0000  0000.jpg  0440  0440.jpg  0960  0960.jpg
    pairs = []
    for sub_item in os.listdir(ours_vid_path):
        if '.jpg' in sub_item:
            src_img_path = os.path.join(ours_vid_path, sub_item)
            tgt_img_path = os.path.join(pG2_vid_path, sub_item)

            # print(src_img_path, tgt_img_path)
            shutil.copy(src_img_path, tgt_img_path)
        else:
            ours_vid_sub_item_path = os.path.join(ours_vid_path, sub_item)
            pG2_vid_sub_item_path = os.path.join(pG2_vid_path, sub_item)

            for sub_sub_item in os.listdir(ours_vid_sub_item_path):
                ours_vid_sub_sub_item_path = os.path.join(ours_vid_sub_item_path, sub_sub_item)
                pG2_vid_sub_sub_item_path = os.path.join(pG2_vid_sub_item_path, sub_sub_item)

                for img in os.listdir(pG2_vid_sub_sub_item_path):

                    if 'gt_' in img:
                        src_img_path = os.path.join(ours_vid_sub_sub_item_path, img)
                        tgt_img_path = os.path.join(pG2_vid_sub_sub_item_path, img)
                        shutil.copy(src_img_path, tgt_img_path)
                        # print(src_img_path, tgt_img_path)

        print(sub_item)
