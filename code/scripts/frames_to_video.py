"""
Converts frames to video.
"""
from argparse import ArgumentParser
from video_utils import frames_to_video

"""
Arguments loading
"""
parser = ArgumentParser()

# The idea here is to save the automatic first frame inside the folder of the dataset
parser.add_argument('--video', default='pigs')
parser.add_argument('--results_folder', help='path to the masks', default='results')
parser.add_argument('--imgs_folder', help='path to the images', default='DAVIS/2017/trainval/JPEGImages/480p')

args = parser.parse_args()

frames_to_video(args.video, args.results_folder, args.imgs_folder)