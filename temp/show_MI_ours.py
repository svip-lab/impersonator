# -*- coding: utf-8 -*-
# @Time    : 2019-03-04 15:06
# @Author  : Zhixin Piao 
# @Email   : piaozhx@shanghaitech.edu.cn

import os
import cv2
import numpy as np
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm


def auto_unzip_fun(x, f):
    return f(*x)


def get_test_actors_result(result_root_path):
    test_actors = os.listdir(result_root_path)

    test_actors_result = []
    for test_actor in test_actors:
        actor_dir_path = os.path.join(result_root_path, test_actor)

        actor_img_name_list = list(filter(lambda x: x.endswith(".jpg"), os.listdir(actor_dir_path)))
        actor_frame_list = list(map(lambda x: x[:-4], actor_img_name_list))

        actor_frames_result = {}
        for actor_frame in actor_frame_list:
            actor_frame_img_path = os.path.join(actor_dir_path, actor_frame + '.jpg')
            actor_frame_dir_path = os.path.join(actor_dir_path, actor_frame)

            # target dir
            target_dir_list = os.listdir(actor_frame_dir_path)
            actor_frame_result = {'frame_img_path': actor_frame_img_path, 'frame_result': {}}

            for target_dir in target_dir_list:
                target_dir_path = os.path.join(actor_frame_dir_path, target_dir)

                img_name_list = sorted(os.listdir(target_dir_path))
                synthesis_img_name_list = list(filter(lambda x: x.find('pred') != -1, img_name_list))
                target_img_name_list = list(filter(lambda x: x.find('gt') != -1, img_name_list))

                actor_frame_result['frame_result'][target_dir] = [
                    {
                        'synthesis_img_path': os.path.join(target_dir_path, synthesis_img_name),
                        'target_img_path': os.path.join(target_dir_path, target_img_name)
                    }
                    for synthesis_img_name, target_img_name in zip(synthesis_img_name_list, target_img_name_list)
                ]

            actor_frames_result[actor_frame] = actor_frame_result

        "result"
        test_actor_result = {
            'actor_dir': actor_dir_path,
            'actor_name': test_actor,
            'actor_frames_result': actor_frames_result
        }
        test_actors_result.append(test_actor_result)

    return test_actors_result


def load_and_comment_images(img_path_list, comments):
    assert len(img_path_list) % 3 == 0

    if len(img_path_list) == 3:
        img_list = []
        for img_path, comment in zip(img_path_list, comments):
            img = cv2.imread(img_path)
            cv2.putText(img, comment, (10, 20), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 0, 255), 1)

            img_list.append(img)

        img = np.concatenate(img_list, axis=1)
    else:
        img_list = []
        for i in range(len(img_path_list) // 3):
            img_list.append(load_and_comment_images(img_path_list[i * 3:(i + 1) * 3], comments[i * 3:(i + 1) * 3]))

        img = np.concatenate(img_list, axis=0)

    return img


def get_args_list(result_root_path):
    test_actors_result = get_test_actors_result(result_root_path)

    args_list = []
    for test_actor_result in test_actors_result:
        actor_name = test_actor_result['actor_name']
        actor_frames_result = test_actor_result['actor_frames_result']

        for actor_frame, actor_frame_result in actor_frames_result.items():
            frame_img_path = actor_frame_result['frame_img_path']
            frame_result = actor_frame_result['frame_result']

            for target_name, synthesis_results in frame_result.items():
                for synthesis_result in synthesis_results:
                    synthesis_img_path = synthesis_result['synthesis_img_path']
                    target_img_path = synthesis_result['target_img_path']

                    args_list.append(([frame_img_path, target_img_path, synthesis_img_path],
                                      [frame_img_path.replace(result_root_path, ''),
                                       target_img_path.replace(result_root_path, ''),
                                       synthesis_img_path.replace(result_root_path, '')]))

    return args_list


def get_args_list_with_2_algos(result_root_path1, result_root_path2):
    test_actors_result1 = get_test_actors_result(result_root_path1)
    test_actors_result2 = get_test_actors_result(result_root_path2)

    args_list = []
    for test_actor_result, test_actor_result_2 in zip(test_actors_result1, test_actors_result2):
        actor_name = test_actor_result['actor_name']
        actor_frames_result = test_actor_result['actor_frames_result']

        for actor_frame, actor_frame_result in actor_frames_result.items():
            frame_img_path = actor_frame_result['frame_img_path']
            frame_result = actor_frame_result['frame_result']

            for target_name, synthesis_results in frame_result.items():
                synthesis_results_2 = test_actor_result_2['actor_frames_result'][actor_frame]['frame_result'][target_name]

                for synthesis_result, synthesis_result_2 in zip(synthesis_results, synthesis_results_2):
                    synthesis_img_path = synthesis_result['synthesis_img_path']
                    target_img_path = synthesis_result['target_img_path']

                    synthesis_img_path_2 = synthesis_result_2['synthesis_img_path']
                    target_img_path_2 = synthesis_result_2['target_img_path']

                    args_list.append(([frame_img_path, target_img_path, synthesis_img_path, frame_img_path, target_img_path_2, synthesis_img_path_2],
                                      ['ours' + frame_img_path.replace(result_root_path1, ''),
                                       'ours' + target_img_path.replace(result_root_path1, ''),
                                       'ours' + synthesis_img_path.replace(result_root_path1, ''),

                                       'shup' + frame_img_path.replace(result_root_path1, ''),
                                       'shup' + target_img_path_2.replace(result_root_path2, ''),
                                       'shup' + synthesis_img_path_2.replace(result_root_path2, '')
                                       ]))

    return args_list


def create_one_dataset_result():
    # result_root_path = '/p300/poseGANs/impersonator_02_21'
    # save_video_name = 'MI_ours'

    result_root_path = '/p300/poseGANs/posewarp/mi_results_10000_preview'
    save_video_name = 'MI_shup_1w_preview'

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fused_w, fused_h = 256 * 3, 256
    save_prefix_path = 'demo'
    fps = 4
    pool_size = 50

    args_list = get_args_list(result_root_path)
    print('args_list ok!')

    """create video"""
    videoWriter = cv2.VideoWriter('%s/%s.avi' % (save_prefix_path, save_video_name), fourcc, fps, (fused_w, fused_h))

    with Pool(pool_size) as p:
        for img in tqdm(p.imap(partial(auto_unzip_fun, f=load_and_comment_images), args_list), total=len(args_list)):
            videoWriter.write(img)
    videoWriter.release()

    save_video_path = os.path.join(save_prefix_path, save_video_name)
    avi_video_path = save_video_path + '.avi'
    mp4_video_path = save_video_path + '.mp4'

    os.system("ffmpeg -y -i %s -vcodec h264 %s" % (avi_video_path, mp4_video_path))


def create_two_dataset_result():
    result_root_path_1 = '/p300/poseGANs/impersonator_02_21'
    result_root_path_2 = '/p300/poseGANs/posewarp/mi_results_10000_preview'
    save_video_name = 'MI_ours_02_21_shup_1w_preview'

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fused_w, fused_h = 256 * 3, 256 * 2
    save_prefix_path = 'demo'
    fps = 4
    pool_size = 50

    args_list = get_args_list_with_2_algos(result_root_path_1, result_root_path_2)
    print('args_list ok!')

    """create video"""
    videoWriter = cv2.VideoWriter('%s/%s.avi' % (save_prefix_path, save_video_name), fourcc, fps, (fused_w, fused_h))

    with Pool(pool_size) as p:
        for img in tqdm(p.imap(partial(auto_unzip_fun, f=load_and_comment_images), args_list), total=len(args_list)):
            videoWriter.write(img)
    videoWriter.release()

    save_video_path = os.path.join(save_prefix_path, save_video_name)
    avi_video_path = save_video_path + '.avi'
    mp4_video_path = save_video_path + '.mp4'

    os.system("ffmpeg -y -i %s -vcodec h264 %s" % (avi_video_path, mp4_video_path))


def get_args_list_with_4_algos(result_root_path1, result_root_path2, result_root_path3, result_root_path4):
    test_actors_result1 = get_test_actors_result(result_root_path1)
    test_actors_result2 = get_test_actors_result(result_root_path2)
    test_actors_result3 = get_test_actors_result(result_root_path3)
    test_actors_result4 = get_test_actors_result(result_root_path4)

    args_list = []
    for test_actor_result, test_actor_result_2, test_actor_result_3, test_actor_result_4 in zip(test_actors_result1, test_actors_result2, test_actors_result3,
                                                                                                test_actors_result4):
        actor_name = test_actor_result['actor_name']
        actor_frames_result = test_actor_result['actor_frames_result']

        for actor_frame, actor_frame_result in actor_frames_result.items():
            frame_img_path = actor_frame_result['frame_img_path']
            frame_result = actor_frame_result['frame_result']

            for target_name, synthesis_results in frame_result.items():
                synthesis_results_2 = test_actor_result_2['actor_frames_result'][actor_frame]['frame_result'][target_name]
                synthesis_results_3 = test_actor_result_3['actor_frames_result'][actor_frame]['frame_result'][target_name]
                synthesis_results_4 = test_actor_result_4['actor_frames_result'][actor_frame]['frame_result'][target_name]

                for synthesis_result, synthesis_result_2, synthesis_result_3, synthesis_result_4 in zip(synthesis_results, synthesis_results_2,
                                                                                                        synthesis_results_3, synthesis_results_4):
                    synthesis_img_path = synthesis_result['synthesis_img_path']
                    target_img_path = synthesis_result['target_img_path']

                    # 2
                    synthesis_img_path_2 = synthesis_result_2['synthesis_img_path']
                    target_img_path_2 = synthesis_result_2['target_img_path']

                    # 3
                    synthesis_img_path_3 = synthesis_result_3['synthesis_img_path']
                    target_img_path_3 = synthesis_result_3['target_img_path']

                    # 4
                    synthesis_img_path_4 = synthesis_result_4['synthesis_img_path']
                    target_img_path_4 = synthesis_result_4['target_img_path']

                    args_list.append(([frame_img_path, target_img_path,
                                       synthesis_img_path_2, synthesis_img_path, synthesis_img_path_3, synthesis_img_path_4],
                                      [frame_img_path.replace(result_root_path1, ''),
                                       target_img_path.replace(result_root_path1, ''),
                                       'shup_1w_p' + synthesis_img_path_2.replace(result_root_path2, ''),

                                       'ours_02_21' + synthesis_img_path.replace(result_root_path1, ''),
                                       'ours_02_29' + synthesis_img_path_3.replace(result_root_path3, ''),
                                       'ours_02_29_v2' + synthesis_img_path_4.replace(result_root_path4, '')
                                       ]))

    return args_list


def create_four_dataset_result():
    result_root_path_1 = '/p300/poseGANs/impersonator_02_21'
    result_root_path_2 = '/p300/poseGANs/posewarp/mi_results_10000_preview'
    result_root_path_3 = '/p300/poseGANs/impersonator_02_29'
    result_root_path_4 = '/p300/poseGANs/impersonator_02_29_v2'
    save_video_name = 'MI_shup_1w_preview_ours_02_21_29_29_v2'

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fused_w, fused_h = 256 * 3, 256 * 2
    save_prefix_path = 'demo'
    fps = 4
    pool_size = 50

    args_list = get_args_list_with_4_algos(result_root_path_1, result_root_path_2, result_root_path_3, result_root_path_4)
    print('args_list ok!')

    """create video"""
    videoWriter = cv2.VideoWriter('%s/%s.avi' % (save_prefix_path, save_video_name), fourcc, fps, (fused_w, fused_h))

    with Pool(pool_size) as p:
        for img in tqdm(p.imap(partial(auto_unzip_fun, f=load_and_comment_images), args_list), total=len(args_list)):
            videoWriter.write(img)
    videoWriter.release()

    save_video_path = os.path.join(save_prefix_path, save_video_name)
    avi_video_path = save_video_path + '.avi'
    mp4_video_path = save_video_path + '.mp4'

    os.system("ffmpeg -y -i %s -vcodec h264 %s" % (avi_video_path, mp4_video_path))


def main():
    # create_two_dataset_result()
    create_four_dataset_result()

if __name__ == '__main__':
    main()
