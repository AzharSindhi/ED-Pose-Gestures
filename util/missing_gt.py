import cv2
import os
import numpy as np
import json
from pycocotools.coco import COCO

# Global variables for paths
BASE_DIR = 'ablation_studies'
os.makedirs(BASE_DIR, exist_ok=True)
GT_FILE = "/home/woody/iwi5/iwi5197h/stratified_folds_unique_margin_8_n5/fold_4/annotations/person_keypoints_val2017.json"
PRED_EDPOSE_FILE_PATH = "logs/evaluations_edpose_new_split4/edpose_finetune0/all_coco/bbox_predictions.json"
PRED_GESTURE_FILE_PATH = "logs/evaluations_multiruns_stratified_21_07/deformable_full_noextra3/all_coco/bbox_predictions.json"
IMAGE_DIR = '/home/woody/iwi5/iwi5197h/stratified_folds_unique_margin_8_n5/fold_4/val2017'

# New folder for missing ground truth predictions
MISSING_GT_DIR = os.path.join(BASE_DIR, 'missing_gt')
os.makedirs(MISSING_GT_DIR, exist_ok=True)

# Load JSON files
def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def compute_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = w1 * h1
    box2_area = w2 * h2

    union_area = box1_area + box2_area - inter_area
    iou = inter_area / union_area if union_area > 0 else 0
    return iou

def draw_bounding_box(image, annotation_dict, cat2name, color=(0, 255, 0)):
    x, y, w, h = annotation_dict['bbox']
    class_name = cat2name[annotation_dict['category_id']]
    if "score" in annotation_dict:
        class_name += f":{annotation_dict['score']:.2f}"

    image = cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
    (text_width, text_height), baseline = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_DUPLEX, 1, 1)
    cv2.rectangle(image, (int(x), int(y - text_height - 6)), (int(x + text_width), int(y)), (0,0,0), -1)
    image = cv2.putText(image, class_name, (int(x), int(y  - 5)), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255), 1)
    return image

def draw_missing_gt(image_path, cat_mapping, gesture_bboxes, edpose_bboxes, gt_bboxes, output_path):
    # Load the image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Create copies of the image for each model and ground truth
    gesture_img = img.copy()
    edpose_img = img.copy()
    gt_img = img.copy()

    # Draw bounding boxes for gesture model predictions (Orange)
    for bbox in gesture_bboxes:
        gesture_img = draw_bounding_box(gesture_img, bbox, cat_mapping, color=(255, 165, 0))  # Orange

    # Draw bounding boxes for ED-Pose predictions (Red)
    for bbox in edpose_bboxes:
        edpose_img = draw_bounding_box(edpose_img, bbox, cat_mapping, color=(0, 0, 255))  # Red

    # Draw bounding boxes for ground truth (Green)
    for bbox in gt_bboxes:
        gt_img = draw_bounding_box(gt_img, bbox, cat_mapping, color=(255, 0, 0))  # Green

    # Concatenate gesture, edpose, and ground truth images
    concatenated_img = np.hstack((gesture_img, edpose_img, gt_img))

    # Save the image showing missing ground truth predictions
    cv2.imwrite(output_path, cv2.cvtColor(concatenated_img, cv2.COLOR_RGB2BGR))

def process_missing_gt(img_id, cat_mapping, edpose_bboxes, gesture_bboxes, gt_bboxes, image_path, iou_threshold=0.5):
    """
    Process predictions that overlap between the two models but do not overlap with the ground truth.
    Save the images in the missing_gt folder.
    """
    missing_gt_gesture_bboxes = []
    missing_gt_edpose_bboxes = []

    # Loop through gesture predictions and find those that do not overlap with any ground truth but overlap with edpose predictions
    for gesture_bbox in gesture_bboxes:
        # Check if gesture_bbox overlaps with any ground truth box
        gesture_has_gt_overlap = False
        for gt_bbox in gt_bboxes:
            if compute_iou(gesture_bbox['bbox'], gt_bbox['bbox']) >= iou_threshold:
                gesture_has_gt_overlap = True
                break
        
        # If no GT overlap, check overlap with edpose predictions
        if not gesture_has_gt_overlap:
            for edpose_bbox in edpose_bboxes:
                iou_with_edpose = compute_iou(gesture_bbox['bbox'], edpose_bbox['bbox'])
                if iou_with_edpose >= iou_threshold:
                    missing_gt_gesture_bboxes.append(gesture_bbox)
                    missing_gt_edpose_bboxes.append(edpose_bbox)
                    break

    # If there are any matching boxes between gesture and edpose predictions that are not in GT, draw them
    if missing_gt_gesture_bboxes and missing_gt_edpose_bboxes:
        output_path = os.path.join(MISSING_GT_DIR, f'{img_id}.png')
        draw_missing_gt(image_path, cat_mapping, missing_gt_gesture_bboxes, missing_gt_edpose_bboxes, gt_bboxes, output_path)

def perform_missing_gt_analysis(coco_gt, pred_edpose, pred_gesture, threshold=0.1, iou_threshold=0.5):
    # Loop through all images in the ground truth
    for img_id in coco_gt.getImgIds():
        image_info = coco_gt.loadImgs(img_id)[0]
        image_path = os.path.join(IMAGE_DIR, image_info['file_name'])
        cat_mapping = {cat['id']: cat['name'] for cat in coco_gt.loadCats(coco_gt.getCatIds())}

        # Ground truth annotations
        gt_bboxes = [{"bbox": ann['bbox'], "category_id": ann["category_id"]} for ann in coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=[img_id]))]

        # Get predictions for the current image from ED-Pose and Gesture models
        edpose_bboxes = [ann for ann in pred_edpose if ann['image_id'] == img_id and ann['score'] >= threshold]
        gesture_bboxes = [ann for ann in pred_gesture if ann['image_id'] == img_id and ann['score'] >= threshold]

        # Process missing ground truth predictions
        process_missing_gt(img_id, cat_mapping, edpose_bboxes, gesture_bboxes, gt_bboxes, image_path, iou_threshold)

if __name__ == "__main__":
    # Load prediction files for both models
    pred_edpose_file = load_json(PRED_EDPOSE_FILE_PATH)
    pred_gesture_file = load_json(PRED_GESTURE_FILE_PATH)

    # Initialize COCO ground truth
    coco_gt = COCO(GT_FILE)

    # Set the IoU threshold for matching and the confidence threshold
    iou_threshold = 0.5  # Adjust IoU threshold as needed
    score_threshold = 0.03  # Adjust confidence threshold as needed

    # Perform missing ground truth analysis
    perform_missing_gt_analysis(coco_gt, pred_edpose_file, pred_gesture_file, threshold=score_threshold, iou_threshold=iou_threshold)
