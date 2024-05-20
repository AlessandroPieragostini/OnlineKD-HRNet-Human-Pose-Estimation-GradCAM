import os
import json
base_path = os.path.join("data", "babypose", "annotations", "person_keypoints_")
base_out_path =  os.path.join("data", "babypose", "person_detection_results", "results_")
modes = ["train", "val", "test", "dummy_test"] 
for m in modes:
    path = base_path + m + ".json"
    out = []
    with open(path, 'r') as f:
        annots = json.load(f)
        for elem in annots["annotations"]:
            out.append(
                {
                    "bbox" : elem["bbox"],
                    "image_id": elem["image_id"],
                    "category_id": elem["category_id"],
                    "score" : 1
                }
            )
    out_path = base_out_path + m + ".json"
    with open(out_path, 'w') as f:
        json.dump(out, f)
