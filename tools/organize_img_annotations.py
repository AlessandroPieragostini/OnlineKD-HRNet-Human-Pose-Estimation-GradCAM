
import copy
import os
import numpy as np
import shutil
import json
import random
import rotate_img_annotations as ria
import cv2

'''
    images_dict è un dizionario
        key: id immagine
        val: 'train', 'val' o 'test'
'''

img_root = os.path.join(".", "data", "babypose", "images")
anno_root = os.path.join(".", "data", "babypose", "annotations")

li = [3, 4, 5, 6, 7, 11, 14, 15, 16, 17, 20, 21, 22, 24, 25, 26, 30, 34, 36, 37, 39, 42, 43, 66, 73, 74, 76]
images_dict = {}

perc = [0.8, 0.1, 0.1] # train, val, test
modes = ["train", "val", "test"]

template_annotations = {'images': [],'categories': [
        {
            "id": 3,
            "name": "infant",
            "supercategory": "human ",
            "color": "#616ff1",
            "metadata": {},
            "keypoint_colors": [
                "#bf5c4d",
                "#d99100",
                "#4d8068",
                "#0d2b80",
                "#9c73bf",
                "#ff1a38",
                "#bf3300",
                "#736322",
                "#33fff1",
                "#3369ff",
                "#9d13bf",
                "#733941"
            ],
            "keypoints": [
                "right_hand",
                "right_elbow",
                "right_shoulder",
                "left_shoulder ",
                "left_elbow",
                "left_hand",
                "right_foot",
                "right_knee",
                "right_hip",
                "left_hip",
                "left_knee",
                "left_foot"
            ],
            "skeleton": [ [1,2], [2,3], [4,5], [5,6], [7,8], [8,9], [10,11], [11, 12] ]
        }
    ],'annotations':[]}

def build_annotations():
    
    '''
    Costruisce annotazioni per train, validation e test
    a partire da quelle esistenti
    '''
    
    tot_annotations = {}

    for mode in modes:
        tot_annotations[mode] = copy.deepcopy(template_annotations)
    
    for num in li:

        with open( os.path.join(anno_root,  'pz' + str(num) + '.json') ) as json_file:
            f = json.load(json_file)
            for elem in f['images']:
                img_id = elem["id"]
                if img_id in images_dict.keys():
                    new_filename = str(img_id).zfill(12) + ".png"
                    mode = images_dict[ int(img_id) ]
                    elem["file_name"] = new_filename
                    elem['path'] = os.path.join(img_root, mode, new_filename)
                    tot_annotations[mode]["images"].append(elem)
        
            for elem in f["annotations"]:
                img_id = elem["image_id"]
                if img_id in images_dict.keys():
                    mode = images_dict.get( int(img_id) )
                    elem["category_id"] = 3
                    elem["segmentation"] = []
                    tot_annotations[mode]['annotations'].append(elem)
    
    for mode in modes:
        with open (os.path.join(anno_root, f'person_keypoints_{mode}.json'), 'w') as j_file:
            json.dump(tot_annotations[mode],j_file)


def retrieve_annotated_imgs_by_pz():
    '''
    Restituisce un dizionario
    key: numero paziente
    val: array di filename di immagini con almeno un'annotazione
    '''
    ret = {}
    
    for num in li:
        
        with open( os.path.join(anno_root , 'pz' + str(num) + '.json') ) as json_file:
            ret[str(num)] = []
            f = json.load(json_file)
            
            annotated = set()
            
            for elem in f["annotations"]:
                annotated.add(elem["image_id"])

            for elem in f['images']:
                img_id = elem["id"]
                if img_id in annotated:
                    filename = elem['path'].replace("/datasets/", "").replace("baby_pose/","")
                    ret[str(num)].append( (filename, img_id))

    
    return ret

def split_by_pz(files):
    '''
    Splitta i pazienti in train, val e test
    I pazienti sono divisi a priori in 80% train, 10% val e 10% test
    '''
    random.seed(1)
    random.shuffle(li)
    train_valtest = np.split(li, [int(len(li)*perc[0])])
    n_val = len(li)*perc[1] # numero di pazienti nel val set
    
    dt = {}
    dt["train"] = train_valtest[0]
    valtest = np.split(train_valtest[1], [int(np.ceil(n_val))])
    dt["val"] = valtest[0]
    dt["test"] = valtest[1]
    
    for mode in modes:
        for num_pz in dt[mode]:
            pzs = files[str(num_pz)]
            copy_files(pzs,mode)

def split_by_perc(files):
    '''
    Splitta le immagini di ciascun paziente 80 in train, 10 val e 10 test
    '''
    random.seed(1)
    for key, pz_files in files.items():
        random.shuffle(pz_files)
        pzs = np.split(pz_files, [int(len(pz_files)*perc[0]), int(len(pz_files)*(perc[0] + perc[1]))])
        for pz, mode in zip(pzs, modes):
            copy_files(pz, mode)


def copy_files(pz_files, mode):
    '''
    Copia le immagini in input nella cartella di destinazione (train, val, test)
    '''
    dest_dir = os.path.join(img_root, mode)

    for filename, img_id in pz_files:
        f = os.path.join(img_root, filename)
        if os.path.isfile(f):
            new_filename = str(img_id).zfill(12) + ".png"
            new_dir = os.path.join(dest_dir, new_filename)
            images_dict[ int(img_id) ] = mode
            img = cv2.imread(f)
            cv2.imwrite(new_dir, cv2.rotate(img, cv2.ROTATE_180))

def rmake_dir(dir):
    '''
    Elimina e ricrea una folder
    '''
    if os.path.isdir(dir):
        shutil.rmtree(dir)
    os.mkdir(dir)

if __name__ == "__main__":
    
    # print("analyzing annotations ...")
    files = retrieve_annotated_imgs_by_pz()
    images_dict = {}
    for mode in modes:
        rmake_dir( os.path.join(img_root, mode) )
    # print("splitting and rotating images ...")
    #split_by_perc(files)
    split_by_pz(files)
    # print("building new annotations ...")
    build_annotations()
    #rotatating annotations
    # print("adjusting annotations ...")
    ria.rotate_annotations(anno_root, modes)
    
    ########################
    #ria.visualize_random_annotations(img_root, anno_root, "train")