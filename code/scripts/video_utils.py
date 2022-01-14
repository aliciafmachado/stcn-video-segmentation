"""
Util functions for converting frames to video and vice versa.
"""
import cv2
import os

def frames_to_video(video, results_path, imgs_path):
    """
    Converts frames to video and save it into current folder.

    Args:
        video: (str) which video we are working with


    """
    imgs_folder = os.path.join(imgs_path, video)
    masks_folder = os.path.join(results_path, video)

    frames_names = [frame for frame in sorted(os.listdir(imgs_folder)) if frame.endswith(".jpg")]
    masks_names = [mask for mask in sorted(os.listdir(masks_folder)) if mask.endswith(".png")]
    
    frame_path = cv2.imread(os.path.join(imgs_folder, frames_names[0]))
    height, width, _ = frame_path.shape

    video = cv2.VideoWriter(video + '.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (width,height))

    for frame_name, mask_name in zip(frames_names, masks_names):
        frame = cv2.imread(os.path.join(imgs_folder, frame_name))
        mask = cv2.imread(os.path.join(masks_folder, mask_name))

        mask_frame = cv2.addWeighted(frame, 1, mask, 1, 0)
        video.write(mask_frame)

    cv2.destroyAllWindows()
    video.release()