"""
File that calls the model of your preference to segmentate the first frame and
that saves it into the output folder we want.
"""

from argparse import ArgumentParser
import numpy as np
import pickle
from os import path
import os
from PIL import Image
from seg_detectron2 import get_mask_detectron2
import cv2

"""
Arguments loading
"""
parser = ArgumentParser()
# TODO: take dataset into consideration
# The idea here is to save the automatic first frame inside the folder of the dataset
parser.add_argument('--real_path', default='../DAVIS/2017/trainval')
parser.add_argument('--pred_path', default='../DAVIS/2017/trainval')
parser.add_argument('--seg_algo', help='swin-transformer / mask-r-cnn / detectron2', default='detectron2')
parser.add_argument('--limit_annotations', help='True / False', default=True)
parser.add_argument('--threshold', help='interval [0,1]', default=0.7)
parser.add_argument('--dataset', help= 'davis2017 or davis2016', default='davis2017')

args = parser.parse_args()

imgs_path = args.real_path
pred_path = args.pred_path
seg_algo = args.seg_algo
limit_annotations = args.limit_annotations
threshold = args.threshold
dataset = args.dataset

get_masks = {
    'detectron2': get_mask_detectron2,
}

# TODO: discover where to save the first frames, so that it is easily accessible with the current code 
# TODO: discover how to call the segmentation algorithms

# We will save our annotations into a new folder called Auto_Annotations
# List the directories in real path / ImageSets and create them in a new "Auto_Annotations" folder
# Create folder if it doesn't exist yet
pred_path = path.join(pred_path, 'Auto_Annotations')
if not path.exists(pred_path):
    os.makedirs(pred_path)
    os.makedirs(path.join(pred_path, '480p'))

pred_path = path.join(pred_path, '480p')
anns_path = path.join(imgs_path, 'Annotations', '480p')

# TODO: 480p or 1080p ???
imgs_path = path.join(imgs_path, 'JPEGImages', '480p')
vid_list = sorted(os.listdir(imgs_path))

# TODO: temporary solution
# Remove .DStore files in macbook
for v in vid_list:
    if v.startswith('.'):
        vid_list.remove(v)

print('Segmenting first frames...')

# Do the prediction for each first frame
get_palette = True

for vid in vid_list:
    print("Processing " + vid + "!")
    # Find paths for the first frame and the annotations
    pred_vid = path.join(pred_path, vid)

    # Find where to save the annotations calculated by the chosen algorithm
    img_path = path.join(imgs_path, vid, '00000.jpg')
    ann_path = path.join(anns_path, vid, '00000.png')

    if not path.exists(pred_vid):
        os.makedirs(pred_vid)

    # Read the frames from original image and true annotations
    # We use the true annotations to identify the same annotations
    # so that we can compare the results with manually
    # annotated frames
    gd_annotations = np.array(Image.open(ann_path).convert("P"))
    orig_img = cv2.imread(img_path)

    # We only need to take the palette once
    if get_palette and dataset == 'davis2017':
        palette = Image.open(path.expanduser(ann_path)).getpalette()
        print(palette)
        print(ann_path)
        get_palette = False

    # Do the prediction using the algorithm selected
    masks_arr = get_masks[seg_algo](orig_img, gd_annotations, threshold=threshold, 
                                    limit_annotations=limit_annotations, dataset=dataset)

    # Save the first frame into the folder
    mask = Image.fromarray(masks_arr).convert("P")
    if dataset == 'davis_2017':
      mask.putpalette(palette)

    mask.save(path.join(pred_vid, '00000.png'))

print('Finished!')