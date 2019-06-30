import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os


# set matplotlib parameters
plt.rcParams['animation.ffmpeg_path'] ='C:\\ffmpeg\\bin\\ffmpeg.exe'


# frame index
t = 0
# video index
show_v = 0
# toydata
# threshold = 0.4
# videos = [-8, -8, -7, -1, -2, -3, -4, -5, -6]
# total_videos = len(videos)
# formated_data = format_video_data('D:\\0574CrayfishVisualizeData\\toydata\\1_model.ckpt-15000',
#                                   'D:\\0574CrayfishVisualizeData\\toydata\\model.ckpt-15000')

# threshold = 0.5
# videos = [1, 1, 2, 3, 4, 5, 6]
# total_videos = len(videos)
# formated_data = format_video_data('D:\\0574CrayfishVisualizeData\\ped2\\1_model.ckpt-56000',
#                                   'D:\\0574CrayfishVisualizeData\\ped2\\model.ckpt-56000')

threshold = 0.6
videos = [2, 3, 4]
total_videos = len(videos)
formated_data = format_video_data('D:\\0574CrayfishVisualizeData\\avenue\\1_model.ckpt-28000',
                                  'D:\\0574CrayfishVisualizeData\\avenue\\model.ckpt-28000')

black = np.zeros((256, 256, 3), dtype=np.uint8)

dataset = formated_data['dataset']
print(dataset)
images_paths = formated_data['images_paths']
preds_paths = formated_data['preds_paths']
gts = formated_data['gt']
psnrs = formated_data['psnr']
scores = formated_data['scores']
videos_names = formated_data['names']


def load_images(images_list, index):
    length = len(images_list)
    if index == length - 1:
        return None
    else:
        image = cv2.imread(images_list[index], cv2.COLOR_BGR2RGB)
        return image


def load_gt_pred_diff(video_idx, frame_index, normal=True):
    global images_paths, preds_paths
    gt_frame = cv2.imread(images_paths[video_idx][frame_index])
    pred_frame = cv2.imread(preds_paths[video_idx][frame_index])

    cv2.cvtColor(gt_frame, cv2.COLOR_BGR2RGB, gt_frame)
    cv2.cvtColor(pred_frame, cv2.COLOR_BGR2RGB, pred_frame)
    diff_frame = np.abs(gt_frame - pred_frame)
    if normal:
        diff_frame[diff_frame > 0.6 * diff_frame.max()] = 0
    else:
        diff_frame[diff_frame > 0.9 * diff_frame.max()] = 0

    return gt_frame, pred_frame, diff_frame


fig = plt.figure()
ax1 = fig.add_subplot(2, 3, 1)  # gt frame axis
ax2 = fig.add_subplot(2, 3, 2)  # pred frame axis
ax3 = fig.add_subplot(2, 3, 3)  # diff frame axis
ax4 = fig.add_subplot(2, 1, 2)  # score axis
ax5 = fig.add_subplot(2, 3, 6)  # labels axis

# ax1, gt frame axis
ax1.set_xlabel(r'$Gt_t$', fontsize=15)
ax1.set_xticks([]), ax1.set_yticks([])
ax1_image_handel = ax1.imshow(black, animated=True)

# ax2, pred frame axis
ax2.set_xlabel(r'$Pred_t$', fontsize=15)
ax2.set_xticks([]), ax2.set_yticks([])
ax2_image_handel = ax2.imshow(black, animated=True)
ax2.set_title('Testing on ' + dataset, fontsize=30)

# ax3, diff frame axis
ax3.set_xlabel(r'$|Gt_t - Pred_t|$', fontsize=15)
ax3.set_xticks([]), ax3.set_yticks([])
ax3_image_handel = ax3.imshow(black, animated=True)

# ax4, scroes frame axis
show_number = 1500
ax4.set_xlabel('#Frame(t)', fontsize=15)
ax4.set_ylabel('Score', fontsize=15)
ax4.set_ylim(0, 1.0)
ax4.set_xlim(0, show_number)
ax4_line_handel, = ax4.plot([], [], animated=True)

# ax5, labels axis
ax5.plot([], [])
ax5.set_xlim(0, 8)
ax5.set_ylim(0, 1.0)
ax5.axis('off')
gt_label_text = ax5.text(3, 0.9, '', fontsize=15)
pred_label_text = ax5.text(3, 0.8, '', fontsize=15)
video_label_text = ax5.text(3, 0.7, '', fontsize=15)


# set initial state of the animator function
def init_func():
    pass


def update_func(*args):
    global t, show_v, total_videos, videos
    if show_v < total_videos:
        v = videos[show_v]
        length = len(images_paths[v])
        if t < length:
            gt_label = 'Abnormal' if gts[v][t] else 'Normal'
            pred_label = 'Abnormal' if scores[v][t] < threshold else 'Normal'
            video_name = videos_names[v]
            gt_frame, pred_frame, diff_frame = load_gt_pred_diff(v, t, 1 - gts[v][t])
            t += 1
            factor = int(t / show_number)
            start = factor * show_number
            end = (factor + 1) * show_number
            x_data = np.arange(t % show_number)
            score_data = scores[v][start:t]
        else:
            t = 0
            show_v += 1
            gt_frame, pred_frame, diff_frame = black, black, black
            x_data = []
            score_data = []
            gt_label = 'Normal'
            pred_label = 'Normal'
            video_name = ''

        ax1_image_handel.set_array(gt_frame)
        ax2_image_handel.set_array(pred_frame)
        ax3_image_handel.set_array(diff_frame)
        ax4_line_handel.set_data(x_data, score_data)
        gt_label_text.set_text(r'True Label: ' + gt_label)
        pred_label_text.set_text(r'Pred Label: ' + pred_label)
        video_label_text.set_text(r'video: ' + video_name)
    # else:
    #     show_v = 0
    return ax1_image_handel, ax2_image_handel, ax3_image_handel, ax4_line_handel, \
           gt_label_text, pred_label_text, video_label_text


ani = animation.FuncAnimation(fig, update_func, interval=1, blit=True)
# ani.save('dynamic_images.mp4', fps=30)
plt.show()
