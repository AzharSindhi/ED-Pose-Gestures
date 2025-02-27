import json
import os
import cv2
from tqdm import tqdm
from skimage import io
from pycocotools.coco import COCO


def check_all_imgs(coco_json_path, images_dir):
    # check if all the images in json have exactly channels
    with open(coco_json_path, "r") as f:
        data = json.load(f)
    corrupt_filenames = []
    for image_info in tqdm(data["images"]):
        image_path = os.path.join(images_dir, image_info["file_name"])

        try:
            img = io.imread(image_path)
            if img.shape[2]!=3:
                raise (f"have no 3 channels, {img.shape}")
            # Do stuff with img
        except Exception as e:
            print(f"{image_info['file_name']} caused {e}")
            corrupt_filenames.append(image_info["file_name"])
    return corrupt_filenames


def remove_corrupt_annotations(corrupt_filenames_path, incoco_json_path, out_json_path):
    with open(corrupt_filenames_path, "r") as f:
        filenames_to_remove = f.read().strip().splitlines()
    if len(filenames_to_remove) == 0:
        print("Nothing to remove, not creating output file")
        return
    
    incoco = COCO(incoco_json_path)
    
    img_ids_to_keep = []
    for image_info in tqdm(incoco.dataset["images"]):
        if image_info["file_name"] in filenames_to_remove:
            continue
        img_ids_to_keep.append(image_info["id"])
    
    ann_ids = incoco.getAnnIds(imgIds=img_ids_to_keep)
    images = incoco.loadImgs(ids=img_ids_to_keep)
    annotations = incoco.loadAnns(ids=ann_ids)

    incoco.dataset["images"] = images
    incoco.dataset["annotations"] = annotations

    with open(out_json_path, "w") as f:
        json.dump(incoco.dataset, f)



if __name__ == "__main__":
    # coco_json_path = "/home/atuin/b193dc/b193dc14/mywork/datasets/deArt_coco/annotations/person_keypoints_val2017.json"
    # images_dir = "/home/atuin/b193dc/b193dc14/mywork/datasets/deArt_coco/val2017"

    # corrupt_filenames = check_all_imgs(coco_json_path, images_dir)

    # with open(os.path.join(images_dir, "../", "val_corrupt_filenames.txt"), "w") as f:
    #     for filename in corrupt_filenames:
    #         f.write(filename + "\n")

    in_coco_json_path = "/home/atuin/b193dc/b193dc14/mywork/datasets/deArt_coco/annotations/person_keypoints_train2017.json"
    out_coco_json_path = "/home/atuin/b193dc/b193dc14/mywork/datasets/deArt_coco/annotations/person_keypoints_train2017_new.json"
    corrupt_filenames_path = "/home/atuin/b193dc/b193dc14/mywork/datasets/deArt_coco/train_corrupt_filenames.txt"
    remove_corrupt_annotations(corrupt_filenames_path, in_coco_json_path, out_coco_json_path)