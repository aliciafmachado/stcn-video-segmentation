"""
Here, we evaluate the results by calculating
the Jaccard score and the F-score for segmentation.
TODO: how do we evaluate when we have different segmentations? 
Do we evaluate each of them separetely?
"""
import numpy as np
import pickle
from os import path
import os
from PIL import Image


def evaluate_results(pred_path, real_path, out_path, imset=None):
    """
    Evaluates the results by calculating the Jaccard score and the F-score for segmentation.
    Args:
        pred_path: (str) path to predicted segmentation masks
        real_path: (str) path to real segmentation masks
        out_path: (str) path where to save the computed metrics
    """
    print("Evaluating results...")

    # Create folder if it doesn't exist yet
    if not path.exists(out_path):
        os.makedirs(out_path)

    # TODO: check if everything here is correct
    # TODO: 480p or 1080p ???
    real_mask_dir = path.join(real_path, 'Annotations', '480p')
    # real_vid_list = sorted(os.listdir(real_mask_dir))

    pred_mask_dir = pred_path
    pred_vid_list = sorted(os.listdir(pred_mask_dir))

    # TODO: temporary solution
    # Remove .DStore files in macbook
    for v in pred_vid_list:
        if v.startswith('.'):
            pred_vid_list.remove(v)

    # imset_path = path.join(real_path, 'ImageSets', imset)

    jacc = 0
    f_score = 0
    n_vid = len(pred_vid_list)
    # videos = []

    # This will work only for davis dataset 2017
    # TODO: make it work for YoutubeVOS and davis 2016 dataset
    # with open(imset_path, "r") as lines:
    #         for line in lines:
    #             _video = line.rstrip('\n')
    #             videos.append(_video)
    #             self.num_frames[_video] = len(os.listdir(path.join(self.image_dir, _video)))
    #             _mask = np.array(Image.open(path.join(self.mask_dir, _video, '00000.png')).convert("P"))
    #             self.num_objects[_video] = np.max(_mask)
    #             self.shape[_video] = np.shape(_mask)
    #             _mask480 = np.array(Image.open(path.join(self.mask480_dir, _video, '00000.png')).convert("P"))
    #             self.size_480p[_video] = np.shape(_mask480)

    for i in range(n_vid):
        real_mask_path_i = path.join(real_mask_dir, pred_vid_list[i])
        pred_mask_path_i = path.join(pred_mask_dir, pred_vid_list[i])

        # TODO: load folder of videos
        frames_video = sorted(os.listdir(pred_mask_path_i))

        # TODO: temporary solution
        # Remove .DStore files in macbook
        for v in frames_video:
            if v.startswith('.'):
                pred_vid_list.remove(v)

        # TODO: Do we just change into a boolean??
        cur_jacc, cur_f_score = eval_metrics(pred_mask_path_i, real_mask_path_i, frames_video)
        jacc += cur_jacc / n_vid
        f_score += cur_f_score / n_vid

    print("Jaccard score: {}".format(jacc))
    print("F-score: {}".format(f_score))
    save_metrics({'jacc': jacc, 'f_score': f_score}, out_path)

    print("Finished evaluating results!")


def save_metrics(metrics, out_path):
    """
    Saves the metrics to a pickle file.
    Args:
        metrics: (dict) dictionary containing the metrics
        out_path: (str)
    """
    with open(path.join(out_path, 'metrics.pickle'), 'wb') as f:
        pickle.dump(metrics, f)


def f_score(pred_mask, real_mask, beta=1):
    """
    Calculates f-score for a single image.
    TODO: Correct this function !!!
    Args:
        pred_mask: (H, W)
        gt_mask: (H, W)
    Returns:
        score: (1,)
    """
    pred_mask = pred_mask.astype(np.bool)
    real_mask = real_mask.astype(np.bool)

    intersection = np.sum(pred_mask * real_mask)
    union = np.sum(pred_mask) + np.sum(real_mask) - intersection
    score = (1 + beta ** 2) * intersection / (beta ** 2 * union + intersection)

    return score


def eval_metrics(pred_mask, real_mask, frames_video):
    """
    Calculate the Jaccard and F scores for a given segmentation mask.
    Args:
        pred_mask: (N, H, W)
        real_mask: (N, H, W)
    Returns:
        jacc: (N,)
        f_score: (N,)
    """
    # pred_mask = pred_mask.cpu().numpy()
    # gt_mask = real_mask.cpu().numpy()

    # pred_mask = np.squeeze(pred_mask)
    # gt_mask = np.squeeze(real_mask)

    jacc = []
    f_scores = []

    for i in range(len(frames_video)):
        # Read the frame and transform it into (H, W):
        pred_frame = np.array(Image.open(path.join(pred_mask, frames_video[i])).convert("P"))
        real_frame = np.array(Image.open(path.join(real_mask, frames_video[i])).convert("P"))

        jacc.append(jaccard(pred_frame, real_frame))
        f_scores.append(f_score(pred_frame, real_frame))
    
    return np.mean(jacc), np.mean(f_scores)


def jaccard(pred_mask, real_mask):
    """
    Calculates jaccard score for a single image.
    TODO: we will start implementing it by taking bits that ain't zero are 1
    Args:
        pred_mask: (H, W)
        real_mask: (H, W)
    Returns:
        score: (1,)
    """
    pred_mask = pred_mask.astype(bool)
    real_mask = real_mask.astype(bool)

    # intersection = np.sum(np.isclose(pred_mask, real_mask).astype(int) * np.invert(np.isclose(
    #     pred_mask, 0)).astype(int) * np.invert(np.isclose(real_mask, 0)).astype(int))
    # union = np.sum((np.invert(np.isclose(real_mask, 0))).astype(int)) + np.sum(
    #     (np.invert(np.isclose(real_mask, 0))).astype(int)) - intersection

    intersection = np.sum(pred_mask * real_mask)
    union = np.sum(pred_mask) + np.sum(real_mask) - intersection

    score = intersection / union
    return score 