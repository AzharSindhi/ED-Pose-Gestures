import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import cv2
import matplotlib.patches as patches

# Global variables for paths
BASE_DIR = 'ablation_studies'
os.makedirs(BASE_DIR, exist_ok=True)
GT_FILE = "/home/woody/iwi5/iwi5197h/stratified_folds_unique_margin_8_n5/fold_4/annotations/person_keypoints_val2017.json"
PRED_EDPOSE_FILE_PATH = "logs/evaluations_edpose_new_split4/edpose_finetune0/all_coco/bbox_predictions.json"
PRED_GESTURE_FILE_PATH = "logs/evaluations_multiruns_stratified_21_07/deformable_full_noextra3/all_coco/bbox_predictions.json"
IMAGE_DIR = '/home/woody/iwi5/iwi5197h//stratified_folds_unique_margin_8_n5/fold_4/val2017'

# Load JSON files
def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def save_results(results, directory, file_name):
    with open(os.path.join(directory, file_name), 'w') as f:
        json.dump(results, f, indent=4)

def create_folder_structure(base_dir):
    folders = ['better_performance', 'worse_performance', 'results_with_threshold', 
               'best_threshold', 'qualitative_analysis', 'better_prediction_images', 
               'worse_prediction_images', 'correct_predictions_by_model', 
               'correct_predictions_by_edpose']
    for folder in folders:
        os.makedirs(os.path.join(base_dir, folder), exist_ok=True)

def evaluate_model(coco_gt, pred_file, threshold):
    filtered_preds = [pred for pred in pred_file if pred['score'] >= threshold]
    
    if len(filtered_preds) == 0:  # Handle empty predictions
        return 0.0
    
    temp_pred_file = "temp_filtered_preds.json"
    with open(temp_pred_file, 'w') as f:
        json.dump(filtered_preds, f)
    
    coco_pred = coco_gt.loadRes(temp_pred_file)
    
    coco_eval = COCOeval(coco_gt, coco_pred, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    os.remove(temp_pred_file)
    
    return coco_eval.stats[0]

def get_best_threshold(coco_gt, pred_file):
    best_threshold = 0
    best_mAP = 0
    
    thresholds = []
    mAP_scores = []
    
    for threshold in np.arange(0.01, 0.3, 0.01):
        mAP = evaluate_model(coco_gt, pred_file, threshold)
        
        thresholds.append(threshold)
        mAP_scores.append(mAP)
        
        if mAP > best_mAP:
            best_mAP = mAP
            best_threshold = threshold
    
    return best_threshold, best_mAP, thresholds, mAP_scores

def get_top_n_images(pred_file, n=100):
    # Group predictions by image and sum the scores for each image
    image_scores = {}
    for pred in pred_file:
        img_id = pred['image_id']
        image_scores[img_id] = image_scores.get(img_id, 0) + pred['score']
    
    # Sort images by their cumulative score and select the top n
    top_images = sorted(image_scores.items(), key=lambda x: x[1], reverse=True)[:n]
    return [img_id for img_id, _ in top_images]

def plot_threshold_vs_ap(thresholds, mAP_edpose_scores, mAP_gesture_scores, base_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, mAP_edpose_scores, label='ED-Pose', color='red', marker='o')
    plt.plot(thresholds, mAP_gesture_scores, label='Our Gesture Model', color='blue', marker='o')
    
    plt.xlabel('Threshold')
    plt.ylabel('AP Score')
    plt.title('AP Score vs. Threshold')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(os.path.join(base_dir, 'threshold_vs_ap.png'))
    plt.show()

def visualize_comparison(image_path, edpose_bboxes, gesture_bboxes, gt_bboxes, output_dir, img_id):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 8))
    plt.imshow(img)

    # Plot ED-Pose bounding boxes (Red)
    for bbox in edpose_bboxes:
        x, y, w, h = bbox
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
        plt.gca().add_patch(rect)

    # Plot Gesture Model bounding boxes (Blue)
    for bbox in gesture_bboxes:
        x, y, w, h = bbox
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='b', facecolor='none')
        plt.gca().add_patch(rect)

    # Plot Ground Truth bounding boxes (Green)
    for bbox in gt_bboxes:
        x, y, w, h = bbox
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='g', facecolor='none')
        plt.gca().add_patch(rect)

    plt.title(f'Qualitative Comparison for Image ID: {img_id}')
    plt.axis('off')

    # Saving the visualized image
    output_path = os.path.join(output_dir, f'comparison_{img_id}.png')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def overlay_annotations(image, annotations, output_path, color):
    img = cv2.imread(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(12, 8))
    plt.imshow(img)
    
    for ann in annotations:
        x, y, w, h = ann['bbox']
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none')
        plt.gca().add_patch(rect)
    
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def compare_predictions(coco_gt, pred_edpose, pred_gesture, threshold_edpose, threshold_gesture, base_dir):
    better_prediction_dir = os.path.join(base_dir, 'better_prediction_images')
    worse_prediction_dir = os.path.join(base_dir, 'worse_prediction_images')
    
    correct_by_model_dir = os.path.join(base_dir, 'correct_predictions_by_model')
    correct_by_edpose_dir = os.path.join(base_dir, 'correct_predictions_by_edpose')
    
    for img_id in coco_gt.getImgIds():
        gt_annots = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=[img_id]))
        edpose_annots = [ann for ann in pred_edpose if ann['image_id'] == img_id and ann['score'] >= threshold_edpose]
        gesture_annots = [ann for ann in pred_gesture if ann['image_id'] == img_id and ann['score'] >= threshold_gesture]
        
        correct_edpose = len(edpose_annots)
        correct_gesture = len(gesture_annots)
        
        image_info = coco_gt.loadImgs(img_id)[0]
        image_path = os.path.join(IMAGE_DIR, image_info['file_name'])
        
        if correct_gesture > correct_edpose:
            overlay_annotations(image_path, gesture_annots, os.path.join(better_prediction_dir, f'image_{img_id}_better.png'), 'blue')
        elif correct_edpose > correct_gesture:
            overlay_annotations(image_path, edpose_annots, os.path.join(worse_prediction_dir, f'image_{img_id}_worse.png'), 'red')
        
        # Overlay correctly predicted annotations by our model
        if correct_gesture > 0:
            overlay_annotations(image_path, gesture_annots, os.path.join(correct_by_model_dir, f'image_{img_id}_correct_by_model.png'), 'green')
        
        # Overlay correctly predicted annotations by ED-Pose
        if correct_edpose > 0:
            overlay_annotations(image_path, edpose_annots, os.path.join(correct_by_edpose_dir, f'image_{img_id}_correct_by_edpose.png'), 'purple')
        
        # Overlay ground truth annotations
        overlay_annotations(image_path, gt_annots, os.path.join(correct_by_model_dir, f'image_{img_id}_gt.png'), 'orange')

import numpy as np

def perform_qualitative_analysis(coco_gt, pred_edpose, pred_gesture, base_dir, top_n_images, threshold=0.2):
    qualitative_dir = os.path.join(base_dir, 'qualitative_analysis')
    os.makedirs(qualitative_dir, exist_ok=True)
    
    for img_id in top_n_images:
        image_info = coco_gt.loadImgs(img_id)[0]
        image_path = os.path.join(IMAGE_DIR, image_info['file_name'])
        
        # Filter predictions based on the best-found threshold
        edpose_bboxes = [{"bbox": ann['bbox'], "category_name": ann["category_id"], "score": ann["score"]} for ann in pred_edpose if ann['image_id'] == img_id and ann['score'] >= threshold]
        gesture_bboxes = [{"bbox": ann['bbox'], "category_name": ann["category_id"], "score": ann["score"]} for ann in pred_gesture if ann['image_id'] == img_id and ann['score'] >= threshold]
        gt_bboxes = [{"bbox": ann['bbox'], "category_name": ann["category_id"]} for ann in coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=[img_id]))]

        # Create the concatenated image with the visualizations
        output_path = os.path.join(qualitative_dir, f'comparison_{img_id}.png')
        create_concatenated_visualization(image_path, edpose_bboxes, gesture_bboxes, gt_bboxes, output_path)


def create_concatenated_visualization(image_path, edpose_bboxes, gesture_bboxes, gt_bboxes, output_path):
    # Load the image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Create copies of the image for each visualization
    edpose_img = img.copy()
    gesture_img = img.copy()
    gt_img = img.copy()

    # Define font and color for the text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1

    # Draw bounding boxes and text for ED-Pose predictions (Red)
    for bbox in edpose_bboxes:
        x, y, w, h = bbox['bbox']
        category_name = bbox['category_name']
        score = ""#bbox['score']
        cv2.rectangle(edpose_img, (int(x), int(y)), (int(x+w), int(y+h)), (255, 0, 0), 2)
        cv2.putText(edpose_img, f'{category_name}', (int(x), int(y)-5), font, font_scale, (255, 0, 0), font_thickness)

    # Draw bounding boxes and text for Gesture model predictions (Blue)
    for bbox in gesture_bboxes:
        x, y, w, h = bbox['bbox']
        category_name = bbox['category_name']
        score = "" #bbox['score']
        cv2.rectangle(gesture_img, (int(x), int(y)), (int(x+w), int(y+h)), (0, 0, 255), 2)
        cv2.putText(gesture_img, f'{category_name}', (int(x), int(y)-5), font, font_scale, (0, 0, 255), font_thickness)

    # Draw bounding boxes and text for Ground Truth (Green)
    for bbox in gt_bboxes:
        x, y, w, h = bbox['bbox']
        category_name = bbox['category_name']
        cv2.rectangle(gt_img, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)
        cv2.putText(gt_img, f'{category_name}', (int(x), int(y)-5), font, font_scale, (0, 255, 0), font_thickness)

    # Concatenate images horizontally
    concatenated_img = np.hstack((edpose_img, gesture_img, gt_img))

    # Save the concatenated image
    cv2.imwrite(output_path, cv2.cvtColor(concatenated_img, cv2.COLOR_RGB2BGR))


def main():
    create_folder_structure(BASE_DIR)

    # Load files
    pred_edpose_file = load_json(PRED_EDPOSE_FILE_PATH)
    pred_gesture_file = load_json(PRED_GESTURE_FILE_PATH)

    # Initialize COCO ground truth
    coco_gt = COCO(GT_FILE)

    # Find the best thresholds
    # best_threshold_edpose, best_mAP_edpose, thresholds, mAP_edpose_scores = get_best_threshold(coco_gt, pred_edpose_file)
    # best_threshold_gesture, best_mAP_gesture, _, mAP_gesture_scores = get_best_threshold(coco_gt, pred_gesture_file)

    # print(f'Best Threshold for ED-Pose: {best_threshold_edpose} with mAP: {best_mAP_edpose}')
    # print(f'Best Threshold for Gesture Model: {best_threshold_gesture} with mAP: {best_mAP_gesture}')

    # # Plotting the AP scores against thresholds
    # plot_threshold_vs_ap(thresholds, mAP_edpose_scores, mAP_gesture_scores, BASE_DIR)
    
    # Get top 100 images based on prediction scores
    top_n_images = get_top_n_images(pred_gesture_file, n=100)
    
    # Perform qualitative analysis
    perform_qualitative_analysis(coco_gt, pred_edpose_file, pred_gesture_file, BASE_DIR, top_n_images)
    
    # Compare predictions
    # compare_predictions(coco_gt, pred_edpose_file, pred_gesture_file, best_threshold_edpose, best_threshold_gesture, BASE_DIR)

if __name__ == '__main__':
    main()
