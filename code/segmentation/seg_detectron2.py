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

def get_less_annotations(gd_annotations, new_annotations, threshold=0.7, 
                         dataset='davis2017'):
  """
  Returns the annotations that identify the same objects as the true
  annotations.
  Args:
    gd_annotations: ground truth annotations on the data
    new_annotations: new annotations using segmentation algorithm
    threshold: float between 0 and 1
    dataset: (string) indicates which davis dataset
  Returns:
    less_annotations: new annotations but with less objects trying to mimic
                      ground truth annotations
  """
  # We get the number of annotations from tehe ground truth so that we don't
  # take more annotations then the manual annotated ones
  if dataset == 'davis2016':
    gd_annotations[gd_annotations == 256] = 1
  
  nb_annotations = gd_annotations.max()
  nb_new_annotations = 0

  bool_gd = gd_annotations.astype(bool)
  idxs_to_del = []

  for i in range(new_annotations.shape[0]):
    if nb_new_annotations == nb_annotations:
      idxs_to_del.extend(list(range(i, new_annotations.shape[0])))
      break

    total = np.sum(new_annotations[i,:,:].astype(int))
    intersection = np.sum(np.logical_and(bool_gd,
                                      new_annotations[i,:,:]).astype(int))

    if intersection / total < threshold:
      idxs_to_del.append(i)
    else:
      nb_new_annotations += 1
  
  less_annotations = np.delete(new_annotations, idxs_to_del, axis=0)

  return less_annotations


def get_arr_from_bools(annotations, dataset='davis2017'):
    """
    Functions that gets the array of integers representing the
    masks from the annotations.
    Args:
        annotations: (N_objects, H, W) tensor containing annotations from segmentation
                                        algorithm
        dataset: (string) which davis dataset
    Returns:
        img_arr: (H, W) array of integers showing to which class each pixel is from
    """
    img_arr = np.zeros(annotations.shape[1:])

    if dataset == 'davis2017':
      for i in range(len(annotations)-1, -1, -1):
        img_arr[annotations[i]] = i + 1
    else:
      img_arr[annotations[0]] = 255

    return img_arr


def get_mask_detectron2(orig_img, gd_annotations, threshold, 
                                    limit_annotations, dataset, max_nb_objects):
    """
    Function that calls detectron2 to segmentate the first frame of a video.
    Args:
        orig_img: (H, W, 3)
        gd_annotations: (H, W)
        threshold: float
        limit_annotations: bool
        dataset: (string) indicates which davis dataset we are
          working with.
        max_nb_objects: (int) indicates maximum number of objects
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

    if max_nb_objects > 0:
        masks = masks[:max_nb_objects]
    
    elif limit_annotations:
        masks = get_less_annotations(gd_annotations, masks, threshold, dataset, 
                                     max_nb_objects)
    
    auto_mask = get_arr_from_bools(masks, dataset)
    return auto_mask