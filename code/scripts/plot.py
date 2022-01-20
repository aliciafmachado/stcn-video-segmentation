from matplotlib import pyplot as plt
import os
import cv2
from argparse import ArgumentParser


"""
Arguments loading
"""
parser = ArgumentParser()
# The idea here is to save the automatic first frame inside the folder of the dataset
parser.add_argument('--masks_folder', type=str)
parser.add_argument('--imgs_folder', type=str)
parser.add_argument('--save_path', default='image.png', type=str)
parser.add_argument('--max_frames', default=4, type=int)
parser.add_argument('--frequence', default=5, type=int)

args = parser.parse_args()

def plot_frames(masks_folder, imgs_folder, save_path='image.png', max_frames=4, frequence=5):
    """
    Plot the frames of the video for report.
    """
    frames = list(range(0, max_frames * frequence, frequence))

    # Create subplot
    f, axs = plt.subplots(2, max_frames // 2)
    axs = axs.flatten()

    # Plot frames
    for frame, ax in zip(frames, axs):
        fr = cv2.imread(os.path.join(imgs_folder, f'{frame:05d}.jpg'))
        mask = cv2.imread(os.path.join(masks_folder, f'{frame:05d}.png'))

        frame_mask = cv2.addWeighted(fr, 1, mask, 1, 0)

        ax.imshow(frame_mask[:,:,::-1])
        ax.axis('off')

    f.tight_layout()
    f.savefig(save_path)

print("Beginning ...")

plot_frames(args.masks_folder, args.imgs_folder, args.save_path, args.max_frames, args.frequence)

print("Finished!")