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

# Load JSON files
def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def save_results(results, directory, file_name):
    with open(os.path.join(directory, file_name), 'w') as f:
        json.dump(results, f, indent=4)

def non_max_suppression(bboxes, iou_threshold=0.5):
    """
    Apply Non-Maximum Suppression (NMS) to avoid multiple detections of the same object.
    """
    if len(bboxes) == 0:
        return []

    boxes = np.array([bbox['bbox'] for bbox in bboxes])
    confidences = np.array([bbox['score'] for bbox in bboxes])

    indices = cv2.dnn.NMSBoxes(boxes.tolist(), confidences.tolist(), score_threshold=0.0, nms_threshold=iou_threshold)
    if len(indices) > 0:
        indices = indices.flatten()
        return [bboxes[i] for i in indices]
    else:
        return []

def compute_iou(box1, box2):
    """
    Compute the Intersection over Union (IoU) between two bounding boxes.
    """
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

def draw_bounding_box(image, annotation_dict, cat2name, thickness=2, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.5, font_thickness=1, offset=0, color=(0, 255, 0)):
    """
    Draw a bounding box on an image.
    """
    x, y, w, h = annotation_dict['bbox']
    class_name = cat2name[annotation_dict['category_id']]
    if "score" in annotation_dict:
        class_name += f":{annotation_dict['score']:.2f}"

    x += offset
    y += offset
    image = cv2.rectangle(image, (int(x), int(y )), (int(x+w), int(y + h)), color, thickness)
    (text_width, text_height), baseline = cv2.getTextSize(class_name, font, font_scale, font_thickness)
    cv2.rectangle(image, (int(x), int(y - text_height - 10)), (int(x + text_width), int(y)), (0,0,0), -1)
    image = cv2.putText(image, class_name, (int(x), int(y  - 5)), font, font_scale, (255,255,255), font_thickness)
    return image



def create_concatenated_visualization(image_path, cat_mapping, edpose_bboxes, gesture_bboxes, gt_bboxes, output_path):
    # Load the image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Create copies of the image for each visualization
    edpose_img = img.copy()
    gesture_img = img.copy()
    gt_img = img.copy()

    # Define font and color for the text
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 1
    font_thickness = 2
    thickness = 4
    font_color = (255, 255, 255)

    # Draw bounding boxes and text for ED-Pose predictions (Red)
    for bbox_correct, bbox_conf in edpose_bboxes:
        # print(bbox_correct)
        edpose_img = draw_bounding_box(edpose_img, bbox_correct, cat_mapping, thickness, font, font_scale, font_thickness, color=(0, 0, 255), offset=0)
        # edpose_img = draw_bounding_box(edpose_img, bbox_conf, cat_mapping, thickness, font, font_scale, font_thickness, color=(165, 255, 0), offset=20)

    # Draw bounding boxes and text for Gesture model predictions (Blue)
    for bbox_correct, bbox_conf in gesture_bboxes:
        gesture_img = draw_bounding_box(gesture_img, bbox_correct, cat_mapping, thickness, font, font_scale, font_thickness, offset=0, color=(255, 165, 0))
        # gesture_img = draw_bounding_box(gesture_img, bbox_conf, cat_mapping, thickness, font, font_scale, font_thickness, color=(165, 255, 0), offset=20)

    # Draw bounding boxes and text for Ground Truth (Green)
    for bbox in gt_bboxes:
        gt_img = draw_bounding_box(gt_img, bbox, cat_mapping, thickness, font, font_scale, font_thickness, color=(255, 0, 0))


    # Concatenate images horizontally
    concatenated_img = np.hstack((gesture_img, edpose_img, gt_img))

    # Save the concatenated image
    cv2.imwrite(output_path, cv2.cvtColor(concatenated_img, cv2.COLOR_RGB2BGR))

def perform_qualitative_analysis(coco_gt, pred_edpose, pred_gesture, base_dir, threshold=0.1, iou_threshold=0.5):
    # Create directories for saving qualitative results
    correct_ours_incorrect_edpose_dir, incorrect_ours_correct_edpose_dir, both_wrong_dir, both_correct_dir = create_directories(base_dir)

    # Loop through all images in the ground truth
    for img_id in coco_gt.getImgIds():
        image_info, cat_mapping, image_path, gt_bboxes = load_image_and_gt(coco_gt, img_id)

        # Get predictions from ED-Pose and our model
        edpose_bboxes_correct, gesture_bboxes_correct = get_filtered_predictions(pred_edpose, pred_gesture, gt_bboxes, img_id, threshold, iou_threshold)

        # Compare predictions and process for both scenarios
        process_scenario(img_id, cat_mapping, edpose_bboxes_correct, gesture_bboxes_correct, gt_bboxes, both_correct_dir, image_path, "both_correct", iou_threshold)
        process_scenario(img_id, cat_mapping, edpose_bboxes_correct, gesture_bboxes_correct, gt_bboxes, correct_ours_incorrect_edpose_dir, image_path, "ours_correct_edpose_incorrect", iou_threshold)
        process_scenario(img_id, cat_mapping, edpose_bboxes_correct, gesture_bboxes_correct, gt_bboxes, incorrect_ours_correct_edpose_dir, image_path, "ours_incorrect_edpose_correct", iou_threshold)
        process_scenario(img_id, cat_mapping, edpose_bboxes_correct, gesture_bboxes_correct, gt_bboxes, both_wrong_dir, image_path, "both_wrong", iou_threshold)

# Helper functions

def create_directories(base_dir):
    correct_ours_incorrect_edpose_dir = os.path.join(base_dir, 'correct_ours_incorrect_edpose')
    incorrect_ours_correct_edpose_dir = os.path.join(base_dir, 'incorrect_ours_correct_edpose')
    both_wrong_dir = os.path.join(base_dir, 'both_wrong')
    both_correct_dir = os.path.join(base_dir, 'both_correct')
    os.makedirs(both_correct_dir, exist_ok=True)
    os.makedirs(both_wrong_dir, exist_ok=True)
    os.makedirs(correct_ours_incorrect_edpose_dir, exist_ok=True)
    os.makedirs(incorrect_ours_correct_edpose_dir, exist_ok=True)

    return correct_ours_incorrect_edpose_dir, incorrect_ours_correct_edpose_dir, both_wrong_dir, both_correct_dir


def load_image_and_gt(coco_gt, img_id):
    image_info = coco_gt.loadImgs(img_id)[0]
    image_path = os.path.join(IMAGE_DIR, image_info['file_name'])
    cat_id_to_name = {cat['id']: cat['name'] for cat in coco_gt.loadCats(coco_gt.getCatIds())}
    # Ground truth annotations
    gt_bboxes = [{"bbox": ann['bbox'], "category_id": ann["category_id"]} for ann in coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=[img_id]))]

    return image_info, cat_id_to_name, image_path, gt_bboxes


def get_filtered_predictions(pred_edpose, pred_gesture, gt_bboxes, img_id, threshold, iou_threshold):
    # Get predictions for the current image and filter them by score threshold
    edpose_predictions = [ann for ann in pred_edpose if ann['image_id'] == img_id and ann['score'] >= threshold]
    gesture_predictions = [ann for ann in pred_gesture if ann['image_id'] == img_id and ann['score'] >= threshold]
    edpose_predictions = non_max_suppression(edpose_predictions, iou_threshold)
    gesture_predictions = non_max_suppression(gesture_predictions, iou_threshold)
    return edpose_predictions, gesture_predictions
    # # Initialize filtered bounding boxes for each model
    # filtered_edpose_bboxes = []
    # filtered_gesture_bboxes = []

    # # Filter predictions for each GT object
    # for gt_bbox in gt_bboxes:
    #     # Get the best match for ED-Pose predictions
    #     best_edpose_bbox, conf_best_edpose_bbox = get_best_match_for_gt(gt_bbox, edpose_predictions, iou_threshold)

    #     # Get the best match for Gesture model predictions
    #     best_gesture_bbox, conf_best_gesture_bbox = get_best_match_for_gt(gt_bbox, gesture_predictions, iou_threshold)

    #     if best_edpose_bbox:
    #         filtered_edpose_bboxes.append([best_edpose_bbox, conf_best_edpose_bbox])
    #     if best_gesture_bbox:
    #         filtered_gesture_bboxes.append([best_gesture_bbox, conf_best_gesture_bbox])

    # return filtered_edpose_bboxes, filtered_gesture_bboxes


def get_best_match_for_gt(gt_bbox, predictions, iou_threshold):
    """
    Finds the best matching prediction for a given GT bounding box.
    - If multiple predictions match the GT class, pick the one with the highest confidence.
    - If no predictions match the GT class, pick the highest-confidence prediction irrespective of class.
    """
    best_match = None
    best_iou = 0
    best_score = 0
    best_other_match = None
    best_other_score = 0

    for pred_bbox in predictions:
        iou = compute_iou(gt_bbox['bbox'], pred_bbox['bbox'])

        if iou >= iou_threshold:
            if pred_bbox['category_id'] == gt_bbox['category_id']:  # Match class
                if pred_bbox['score'] > best_score:
                    best_match = pred_bbox
                    best_score = pred_bbox['score']

            if pred_bbox['score'] > best_other_score:
                best_other_match = pred_bbox
                best_other_score = pred_bbox['score']
    
    if best_match is None:
        best_match = best_other_match
    
    return best_match, best_other_match



def process_scenario(img_id, cat_mapping, edpose_bboxes, gesture_bboxes, gt_bboxes, output_dir, image_path, scenario, iou_threshold):
    # Filter predictions according to the scenario (ours_correct_edpose_incorrect or vice versa)
    filtered_gesture_bboxes, filtered_edpose_bboxes = get_filtered_boxes(edpose_bboxes, gesture_bboxes, gt_bboxes, scenario, iou_threshold)
    # hack to consider the nms
    filtered_gesture_bboxes = [[b,b] for b in filtered_gesture_bboxes]
    filtered_edpose_bboxes = [[b,b] for b in filtered_edpose_bboxes]

    # If we found any valid boxes for this scenario, visualize them
    if filtered_gesture_bboxes and filtered_edpose_bboxes:
        output_path = os.path.join(output_dir, f'{img_id}.png')
        create_concatenated_visualization(image_path, cat_mapping, filtered_edpose_bboxes, filtered_gesture_bboxes, gt_bboxes, output_path)
    


def get_filtered_boxes(edpose_bboxes, gesture_bboxes, gt_bboxes, scenario, iou_threshold):
    correct_filtered_gesture_bboxes = []
    wrong_filtered_edpose_bboxes = []

    wrong_filtered_gesture_bboxes = []
    correct_filtered_edpose_bboxes = []

    both_wrong_gesture_bboxes = []
    both_wrong_edpose_bboxes = []

    both_correct_gesture_bboxes = []
    both_correct_edpose_bboxes = []

    for gesture_bbox in gesture_bboxes:
        best_match_edpose_bbox, best_iou = get_best_iou_match(gesture_bbox, edpose_bboxes, iou_threshold)

        if best_match_edpose_bbox and best_iou >= iou_threshold:
            gesture_correct = is_correct(gesture_bbox, gt_bboxes, iou_threshold)
            edpose_correct = is_correct(best_match_edpose_bbox, gt_bboxes, iou_threshold)
            # if both correct, continue
            if gesture_correct and edpose_correct:
                both_correct_gesture_bboxes.append(gesture_bbox)
                both_correct_edpose_bboxes.append(best_match_edpose_bbox)
            # Apply the logic based on the scenario
            elif scenario == "ours_correct_edpose_incorrect" and gesture_correct and not edpose_correct:
                correct_filtered_gesture_bboxes.append(gesture_bbox)
                wrong_filtered_edpose_bboxes.append(best_match_edpose_bbox)
            elif scenario == "ours_incorrect_edpose_correct" and not gesture_correct and edpose_correct:
                wrong_filtered_gesture_bboxes.append(gesture_bbox)
                correct_filtered_edpose_bboxes.append(best_match_edpose_bbox)
            else:
                # both wrong
                both_wrong_gesture_bboxes.append(gesture_bbox)
                both_wrong_edpose_bboxes.append(best_match_edpose_bbox)
    
    if scenario == "both_wrong":
        return both_wrong_gesture_bboxes, both_wrong_edpose_bboxes
    elif scenario == "both_correct":
        return both_correct_gesture_bboxes, both_correct_edpose_bboxes
    elif scenario == "ours_correct_edpose_incorrect":
        return correct_filtered_gesture_bboxes, wrong_filtered_edpose_bboxes
    elif scenario == "ours_incorrect_edpose_correct":
        return wrong_filtered_gesture_bboxes, correct_filtered_edpose_bboxes
    


def get_best_iou_match(gesture_bbox, edpose_bboxes, iou_threshold):
    best_iou = 0
    best_match_edpose_bbox = None

    for edpose_bbox in edpose_bboxes:
        # print(len(gesture_bbox), gesture_bbox)
        # print(len(edpose_bbox), edpose_bbox)
        iou = compute_iou(gesture_bbox['bbox'], edpose_bbox['bbox'])
        if iou > best_iou:
            best_iou = iou
            best_match_edpose_bbox = edpose_bbox

    return best_match_edpose_bbox, best_iou

def match_boxes_by_iou(gesture_bboxes, edpose_bboxes, iou_threshold):
    """
    For each gesture_bbox, find the best matching edpose_bbox based on IoU.
    """
    matched_boxes = []
    
    for gesture_bbox in gesture_bboxes:
        best_iou = 0
        best_match_edpose_bbox = None
        
        for edpose_bbox in edpose_bboxes:
            iou = compute_iou(gesture_bbox['bbox'], edpose_bbox['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_match_edpose_bbox = edpose_bbox

        if best_iou >= iou_threshold:
            matched_boxes.append((gesture_bbox, best_match_edpose_bbox, best_iou))
    
    return matched_boxes

def is_correct(pred_bbox, gt_bboxes, iou_threshold):
    """
    Determine if a predicted bounding box is correct by comparing it to ground truth boxes using IoU.
    """
    for gt_bbox in gt_bboxes:
        iou = compute_iou(pred_bbox['bbox'], gt_bbox['bbox'])
        if iou >= iou_threshold and pred_bbox['category_id'] == gt_bbox['category_id']:
            return True
    return False

if __name__ == "__main__":

    # Create the base folder structure for qualitative analysis
    # create_folder_structure(BASE_DIR)

    # Load prediction files for both models
    pred_edpose_file = load_json(PRED_EDPOSE_FILE_PATH)
    pred_gesture_file = load_json(PRED_GESTURE_FILE_PATH)

    # Initialize COCO ground truth
    coco_gt = COCO(GT_FILE)

    # Set the IoU threshold for matching and the confidence threshold
    iou_threshold = 0.5  # Adjust IoU threshold as needed
    score_threshold = 0.00 # Adjust confidence threshold as needed

    # Perform qualitative analysis
    perform_qualitative_analysis(coco_gt, pred_edpose_file, pred_gesture_file, BASE_DIR, threshold=score_threshold, iou_threshold=iou_threshold)
