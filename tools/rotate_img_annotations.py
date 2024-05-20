#from pycocotools.coco import COCO
import os
import json 
import numpy as np
import random
#import skimage.io as io
#import matplotlib.pyplot as plt


def rotate_annotations(anno_root, modes):
    width = 640
    height = 480
    #coords are 0-indexed
    # x' = width - 1 - x
    # y' = height - 1 - y
    for mode in modes:
        f = None
        with open ( os.path.join(anno_root, f'person_keypoints_{mode}.json') , 'r') as json_file:
            f = json.load(json_file)
            for elem in f["annotations"]:
                #rotate keypoints
                x_key = np.array(elem["keypoints"][0::3])
                y_key = np.array(elem["keypoints"][1::3])
                elem["keypoints"][0::3] = np.ndarray.tolist(width - 1 - x_key)
                elem["keypoints"][1::3] = np.ndarray.tolist(height - 1 - y_key)
                '''
                #rotate segmentation
                x_seg = np.array(elem["segmentation"][0::2])
                y_seg = np.array(elem["segmentation"][1::2])
                elem["segmentation"][0::2] = np.ndarray.tolist(width - 1 - x_seg)
                elem["segmentation"][1::2] = np.ndarray.tolist(height - 1 - y_seg)
                '''
                #rotate bbox
                elem["bbox"][0] = width - 1 - elem["bbox"][0] - elem["bbox"][2]
                elem["bbox"][1] = height - 1 - elem["bbox"][1] - elem["bbox"][3]

        with open (os.path.join(anno_root, f'person_keypoints_{mode}.json') , 'w') as json_file:
            json.dump(f, json_file)

'''
def visualize_random_annotations(img_root, anno_root, mode):
    annotation_file = os.path.join(anno_root, f'person_keypoints_{mode}.json')
    coco=COCO(annotation_file)
    img_ids = coco.getImgIds()
    n_samples = 10
    img_selected = random.sample(img_ids, n_samples)
    for img_id in img_selected:
        img = coco.loadImgs(img_id)[0]
        I = io.imread( os.path.join(img_root, mode, img['file_name']) )
        ann_ids = coco.getAnnIds(imgIds=img_id,iscrowd=None)
        anns = coco.loadAnns(ann_ids)
        plt.imshow(I)
        plt.axis('on')
        coco.showAnns(anns, draw_bbox=True )
        plt.savefig(f"test{img_id}.png", bbox_inches='tight', pad_inches=0)
        plt.close(fig='all')
'''