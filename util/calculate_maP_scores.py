from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import os
import numpy as np

def calculate_ap(gt_json, res_json, score_threshold=0.0, use_cats=True, iou_type="bbox"):
    """
    Calculate AP scores using COCO evaluation metrics.

    :param gt_json: Path to the ground truth JSON file.
    :param res_json: Path to the results JSON file (predictions).
    :param score_threshold: Minimum score threshold to consider a prediction.
    :param use_cats: Whether to use category (class) information in evaluation.
    :return: Dictionary of AP scores.
    """
    if not use_cats:
        # somehow coco eval does not work with keypoints and useCats=False
        coco_gt, coco_dt = change_categories_to_person(gt_json, res_json, 1)
        use_cats = True

    else:
        # Load ground truth annotations
        coco_gt = COCO(gt_json)
        # Load detection results
        coco_dt = coco_gt.loadRes(res_json)

    
    # Filter detections based on score threshold
    coco_dt.anns = {k: v for k, v in coco_dt.anns.items() if v['score'] >= score_threshold}

    # Initialize COCO evaluator
    coco_eval = COCOeval(coco_gt, coco_dt, iouType=iou_type)

    # Set useCats flag
    coco_eval.params.useCats = use_cats
    # Run evaluation
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Retrieve the AP scores
    ap_scores = {
        "AP": coco_eval.stats[0],  # AP at IoU=0.50:0.95
        "AP50": coco_eval.stats[1],  # AP at IoU=0.50
        "AP75": coco_eval.stats[2],  # AP at IoU=0.75
        "AP_small": coco_eval.stats[3],  # AP for small objects
        "AP_medium": coco_eval.stats[4],  # AP for medium objects
        "AP_large": coco_eval.stats[5],  # AP for large objects
    }

    return ap_scores

def change_categories_to_person(gt_json, res_json, cat_id):
    """
    Change the category IDs in the ground truth and results JSON files.

    :param gt_json: Path to the ground truth JSON file.
    :param res_json: Path to the results JSON file (predictions).
    :param new_gt_json: Path to save the new ground truth JSON file.
    :param new_res_json: Path to save the new results JSON file.
    :param cat_ids: Dictionary mapping old category IDs to new category IDs.
    """
    # Load ground truth annotations
    with open(gt_json, "r") as f:
        gt_data = json.load(f)

    # Load detection results
    with open(res_json, "r") as f:
        res_data = json.load(f)

    # Change category IDs in ground truth annotations
    for ann in gt_data["annotations"]:
        ann["category_id"] = cat_id

    # Change category IDs in detection results
    for ann in res_data:
        ann["category_id"] = cat_id
    # save to temp files
    new_gt_json = "temp_gt.json"
    new_res_json = "temp_res.json"
    with open(new_gt_json, "w") as f:
        json.dump(gt_data, f)
    
    with open(new_res_json, "w") as f:
        json.dump(res_data, f)

    coco_gt = COCO(new_gt_json)
    coco_dt = coco_gt.loadRes(new_res_json)
    # delete temp files
    os.remove(new_gt_json)
    os.remove(new_res_json)
    return coco_gt, coco_dt

if __name__ == '__main__':

    # Example usage
    folds_dir = "logs/multiruns_sensorytest_gestures"
    folds_results = {}
    score_threshold = 0.0  # Define your score threshold
    use_cats = True  # Set to False to ignore category information in evaluation
    iou_type = "bbox"  # Set to "keypoints" to evaluate keypoint detection
    i = 1
    for dirname in sorted(os.listdir(folds_dir)):
        if dirname.startswith("vanilla"):
            continue
        if dirname.startswith("extratoken"):
            fold_name = "extratoken"
        if dirname.startswith("classifier_full"):
            fold_name = "classifier_full"
        if dirname.startswith("classifier_partial"):
            fold_name = "classifier_partial"
        elif dirname.startswith("edpose"):
            fold_name = "edpose"
        else:
            fold_name = "_".join(dirname.split("_")[:2])
        
        if fold_name not in folds_results:
            folds_results[fold_name] = []
        
        gt_json = '/net/cluster/azhar/datasets/SensoryGestureRecognition/data/sensoryArt_coco/annotations/person_keypoints_test2017.json'  # Replace with your ground truth JSON file path
        if iou_type == "keypoints":
            res_json = os.path.join(folds_dir, dirname, "all_coco", f"keypoints_predictions{i}.json")
        else:
            res_json = os.path.join(folds_dir, dirname, "all_coco", f"bbox_predictions{i}.json")  # Replace with your results JSON file path

        ap_scores = calculate_ap(gt_json, res_json, score_threshold, use_cats, iou_type)["AP"].round(4)
        folds_results[fold_name].append(ap_scores)
        i+=1

    for k, value in folds_results.items():
        print(k, " mean:", np.mean(value)*100)
        print(k, "std:", np.std(value)*100)
