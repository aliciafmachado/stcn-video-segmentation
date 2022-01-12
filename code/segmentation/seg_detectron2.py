"""
Function that calls detectron2 to segmentate the first frame of a video.

This part uses the structure of calling and doing the inference on the detectron2 from tutorial colab 
in https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5
from the official detectron2 repo.
"""

import numpy as np
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

def get_less_annotations(gd_annotations, new_annotations, threshold=0.7):
  """
  Returns the annotations that identify the same objects as the true
  annotations.
  Args:
    gd_annotations: ground truth annotations on the data
    new_annotations: new annotations using segmentation algorithm
    threshold: float between 0 and 1
  Returns:
    less_annotations: new annotations but with less objects trying to mimic
                      ground truth annotations
  """
  bool_gd = gd_annotations.astype(bool)
  idxs_to_del = []

  for i in range(new_annotations.shape[0]):
    total = np.sum(new_annotations[i,:,:].astype(int))
    intersection = np.sum(np.logical_and(bool_gd,
                                      new_annotations[i,:,:]).astype(int))
    if intersection / total < threshold:
      idxs_to_del.append(i)
  
  less_annotations = np.delete(new_annotations, idxs_to_del, axis=0)

  return less_annotations


def get_arr_from_bools(annotations):
    """
    Functions that gets the array of integers representing the
    masks from the annotations.
    Args:
        annotations: (N_objects, H, W) tensor containing annotations from segmentation
                                        algorithm
    Returns:
        img_arr: (H, W) array of integers showing to which class each pixel is from
    """
    img_arr = np.zeros(annotations.shape[1:])
    for i in range(len(annotations)-1, -1, -1):
        img_arr[annotations[i]] = i + 1

    return img_arr


def get_mask_detectron2(orig_img, gd_annotations, threshold, 
                                    limit_annotations):
    """
    Function that calls detectron2 to segmentate the first frame of a video.
    Args:
        orig_img: (H, W, 3)
        gd_annotations: (H, W)
        threshold: float
        limit_annotations: bool
    Returns:
        auto_mask: (H, W)
    """
    # Load the model:
    cfg = get_cfg()

    # Add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)
    
    masks = predictor(orig_img)["instances"].pred_masks.cpu().numpy()

    if limit_annotations:
        masks = get_less_annotations(gd_annotations, masks, threshold)
    
    auto_mask = get_arr_from_bools(masks)
    return auto_mask