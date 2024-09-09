import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from pycocotools.coco import COCO
from sklearn.metrics import f1_score
from tqdm import tqdm
from qualitative_min import compute_iou, load_json
import os
import cv2


# Global variables for paths
BASE_DIR = 'ablation_studies'
os.makedirs(BASE_DIR, exist_ok=True)
GT_FILE = "/home/woody/iwi5/iwi5197h/stratified_folds_unique_margin_8_n5/fold_4/annotations/person_keypoints_val2017.json"
PRED_EDPOSE_FILE_PATH = "logs/evaluations_edpose_new_split4/edpose_finetune0/all_coco/bbox_predictions.json"
PRED_GESTURE_FILE_PATH = "logs/evaluations_multiruns_stratified_21_07/deformable_full_noextra3/all_coco/bbox_predictions.json"
IMAGE_DIR = '/home/woody/iwi5/iwi5197h/stratified_folds_unique_margin_8_n5/fold_4/val2017'
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve
import numpy as np

def plot_class_distribution(coco_gt, class_names):
    """
    Plot the distribution of class labels in the ground truth dataset.
    """
    # Initialize a dictionary to store the count of each class
    class_counts = {cat['id']: 0 for cat in coco_gt.loadCats(coco_gt.getCatIds())}

    # Loop through all annotations in the dataset and count the occurrences of each class
    for img_id in coco_gt.getImgIds():
        annotations = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=[img_id]))
        for ann in annotations:
            class_counts[ann['category_id']] += 1

    # Extract class names and corresponding counts
    class_labels = [class_names[i-1] for i in class_counts.keys()]
    counts = [class_counts[i] for i in class_counts.keys()]

    # Plot the distribution
    plt.figure(figsize=(12, 6))
    plt.bar(class_labels, counts, color='skyblue')
    plt.title('Class Label Distribution in Ground Truth')
    plt.xlabel('Class Labels')
    plt.ylabel('Number of Instances')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Save the plot as an image
    save_path = os.path.join(BASE_DIR, 'class_distribution.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Class distribution plot saved to {save_path}")


def get_filtered_predictions(pred_edpose, pred_gesture, gt_bboxes, img_id, threshold, iou_threshold):
    # Get predictions for the current image and filter them by score threshold
    edpose_predictions = [ann for ann in pred_edpose if ann['image_id'] == img_id and ann['score'] >= threshold]
    gesture_predictions = [ann for ann in pred_gesture if ann['image_id'] == img_id and ann['score'] >= threshold]

    # Apply Non-Maximum Suppression (NMS) for ED-Pose and Gesture model predictions
    filtered_edpose_bboxes = apply_nms(edpose_predictions, iou_threshold)
    filtered_gesture_bboxes = apply_nms(gesture_predictions, iou_threshold)

    return filtered_edpose_bboxes, filtered_gesture_bboxes

def apply_nms(predictions, iou_threshold):
    """
    Apply OpenCV's Non-Maximum Suppression (NMS) to the bounding boxes in predictions.
    """
    if len(predictions) == 0:
        return []

    boxes = [pred['bbox'] for pred in predictions]
    confidences = [pred['score'] for pred in predictions]

    # Convert boxes to OpenCV's expected format: [x, y, x+w, y+h]
    boxes_cv = [[x, y, x+w, y+h] for (x, y, w, h) in boxes]

    # Apply NMS using OpenCV
    indices = cv2.dnn.NMSBoxes(boxes_cv, confidences, score_threshold=0.0, nms_threshold=iou_threshold)

    # Return only the selected bounding boxes based on NMS
    if len(indices) > 0:
        indices = indices.flatten()
        return [predictions[i] for i in indices]
    else:
        return []


def plot_confusion_matrix(coco_gt, pred_edpose, pred_gesture, class_names, threshold=0.1, iou_threshold=0.5):
    """
    Calculate and plot the confusion matrix for both ED-Pose and our model.
    Ensure that the confusion matrix includes all ground truth labels, even if there's no matching prediction.
    """
    # Create a mapping from COCO category IDs to their indices in class_names
    categories = coco_gt.loadCats(coco_gt.getCatIds())
    category_id_to_index = {cat['id']: i for i, cat in enumerate(categories)}
    
    # Add -1 as a class to represent "No Prediction"
    category_id_to_index[-1] = len(class_names)
    
    # y_true and y_pred arrays
    y_true = []
    y_pred_edpose = []
    y_pred_gesture = []

    # Loop through all images in the ground truth
    for img_id in tqdm(coco_gt.getImgIds()):
        # Get ground truth bounding boxes and labels for the image
        gt_bboxes = [{"bbox": ann['bbox'], "category_id": ann["category_id"]} for ann in coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=[img_id]))]

        # Get filtered predictions for both models
        edpose_bboxes, gesture_bboxes = get_filtered_predictions(pred_edpose, pred_gesture, gt_bboxes, img_id, threshold, iou_threshold)

        # Append GT labels and corresponding predictions
        for gt_bbox in gt_bboxes:
            # Append the mapped index for the ground truth category
            y_true.append(category_id_to_index[gt_bbox['category_id']])

            # Find the best match from ED-Pose predictions and Gesture model
            edpose_pred = find_best_prediction_for_gt(gt_bbox, edpose_bboxes, iou_threshold)
            gesture_pred = find_best_prediction_for_gt(gt_bbox, gesture_bboxes, iou_threshold)

            # Append predictions for ED-Pose model
            if edpose_pred:
                y_pred_edpose.append(category_id_to_index[edpose_pred['category_id']])
            else:
                # If no match, append the index for "No Prediction"
                y_pred_edpose.append(category_id_to_index[-1])

            # Append predictions for Gesture model
            if gesture_pred:
                y_pred_gesture.append(category_id_to_index[gesture_pred['category_id']])
            else:
                y_pred_gesture.append(category_id_to_index[-1])

    # Now we plot the confusion matrices
    f1_edpose = f1_score(y_true, y_pred_edpose, average='macro').round(2)
    f1_gesture = f1_score(y_true, y_pred_gesture, average='macro').round(2)

    save_dir = os.path.join(BASE_DIR, "confusion_matrices")
    os.makedirs(save_dir, exist_ok=True)

    save_dir_ed = os.path.join(save_dir, f"cm_edpose_pred{threshold}_iou{iou_threshold}.pdf")
    save_dir_gesture = os.path.join(save_dir, f"cm_gesture_pred{threshold}_iou{iou_threshold}.pdf")

    # Plot confusion matrix for ED-Pose model
    plot_matrix(y_true, y_pred_edpose, class_names, f"ED-Pose Confusion Matrix\nPred threshold: {threshold}, IoU: {iou_threshold}", f1_edpose, save_dir_ed)

    # Plot confusion matrix for Gesture model
    plot_matrix(y_true, y_pred_gesture, class_names, f"Our Model Confusion Matrix\nPred threshold: {threshold}, IoU: {iou_threshold}", f1_gesture, save_dir_gesture)

def find_best_prediction_for_gt(gt_bbox, predictions, iou_threshold):
    """
    Find the best prediction (based on IoU) for the given ground truth bbox.
    """
    best_iou = 0
    best_prediction = None

    for pred_bbox in predictions:
        iou = compute_iou(gt_bbox['bbox'], pred_bbox['bbox'])

        if iou > best_iou and iou >= iou_threshold:
            best_iou = iou
            best_prediction = pred_bbox

    return best_prediction


def plot_matrix(y_true, y_pred, class_names, title, f1_score, save_dir):
    """
    Helper function to plot the confusion matrix with properly displayed axis labels and class names.
    """
    # Create the confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))

    # Set figure size based on the number of class names
    plt.figure(figsize=(10,8))  # Adjust figure size based on class names length

    # Plot the confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)

    # Add titles and labels for GT (Y-axis) and Pred (X-axis)
    plt.title(f"{title}\n\nF1-Score: {f1_score:.3f}")
    plt.xlabel('Predicted Label')
    plt.ylabel('Ground Truth Label')

    # Rotate the x-axis labels for better readability
    # plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels 45 degrees to avoid overlap
    # plt.yticks(rotation=45, ha='left') 
    # Use tight layout to ensure everything fits
    plt.tight_layout()

    # Save the plot as a PDF and PNG
    plt.savefig(save_dir, format='pdf', dpi=600)
    save_dir_png = save_dir.replace('.pdf', '.png')
    plt.savefig(save_dir_png, dpi=300)
    plt.close()






def generate_metrics_at_thresholds(coco_gt, pred_edpose, pred_gesture, class_names, thresholds, base_dir):
    """
    Generate precision, recall, F1-score, and precision-recall curves for given thresholds and save them.
    """
    # Initialize lists to store the metrics for each threshold
    precisions_edpose = []
    recalls_edpose = []
    f1_scores_edpose = []

    precisions_gesture = []
    recalls_gesture = []
    f1_scores_gesture = []
    categories = coco_gt.loadCats(coco_gt.getCatIds())
    category_id_to_index = {cat['id']: i for i, cat in enumerate(categories)}
    
    # Add -1 as a class to represent "No Prediction"
    category_id_to_index[-1] = len(class_names)
    # Loop through thresholds and calculate precision, recall, F1 for each model
    for threshold in thresholds:
        y_true, y_pred_edpose, y_pred_gesture = get_predictions_and_labels(coco_gt, pred_edpose, pred_gesture, threshold, category_id_to_index)

        # Calculate precision, recall, and F1-score for ED-Pose
        precisions_edpose.append(precision_score(y_true, y_pred_edpose, average='macro', zero_division=0))
        recalls_edpose.append(recall_score(y_true, y_pred_edpose, average='macro', zero_division=0))
        f1_scores_edpose.append(f1_score(y_true, y_pred_edpose, average='macro', zero_division=0))

        # Calculate precision, recall, and F1-score for Gesture model
        precisions_gesture.append(precision_score(y_true, y_pred_gesture, average='macro', zero_division=0))
        recalls_gesture.append(recall_score(y_true, y_pred_gesture, average='macro', zero_division=0))
        f1_scores_gesture.append(f1_score(y_true, y_pred_gesture, average='macro', zero_division=0))

    # Plot and save precision-recall curves for each model
    plot_and_save_curve(thresholds, precisions_edpose, recalls_edpose, 'ED-Pose', 'Precision', 'Recall', 'Precision-Recall Curve for ED-Pose', base_dir)
    plot_and_save_curve(thresholds, precisions_gesture, recalls_gesture, 'Gesture Model', 'Precision', 'Recall', 'Precision-Recall Curve for Combined Gesture Model', base_dir)

    # Plot and save F1-score curve
    plot_and_save_f1_curve(thresholds, f1_scores_edpose, f1_scores_gesture, base_dir)

def get_predictions_and_labels(coco_gt, pred_edpose, pred_gesture, threshold, category_id_to_index):
    """
    Get ground truth labels and predictions for ED-Pose and Gesture model at a given threshold.
    Maps category IDs using category_id_to_index.
    """
    y_true = []
    y_pred_edpose = []
    y_pred_gesture = []

    for img_id in coco_gt.getImgIds():
        # Get ground truth bounding boxes and labels for the image
        gt_bboxes = [{"bbox": ann['bbox'], "category_id": ann["category_id"]} for ann in coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=[img_id]))]
        
        # Get predictions for the current image from ED-Pose and Gesture models
        edpose_bboxes, gesture_bboxes = get_filtered_predictions(pred_edpose, pred_gesture, gt_bboxes, img_id, threshold, iou_threshold=0.9)

        # Append GT labels and corresponding predictions for ED-Pose and Gesture models
        for gt_bbox in gt_bboxes:
            # Append the mapped index for the ground truth category
            y_true.append(category_id_to_index[gt_bbox['category_id']])

            # Find best matching prediction for ED-Pose and map its category ID
            edpose_pred = find_best_prediction_for_gt(gt_bbox, edpose_bboxes, iou_threshold=0.9)
            if edpose_pred:
                y_pred_edpose.append(category_id_to_index[edpose_pred['category_id']])
            else:
                # If no match, append -1 for "No Prediction"
                y_pred_edpose.append(category_id_to_index[-1])

            # Find best matching prediction for Gesture model and map its category ID
            gesture_pred = find_best_prediction_for_gt(gt_bbox, gesture_bboxes, iou_threshold=0.9)
            if gesture_pred:
                y_pred_gesture.append(category_id_to_index[gesture_pred['category_id']])
            else:
                y_pred_gesture.append(category_id_to_index[-1])

    return y_true, y_pred_edpose, y_pred_gesture

def plot_and_save_curve(thresholds, precisions, recalls, model_name, ylabel1, ylabel2, title, base_dir):
    """
    Plot the precision-recall curve and save it to a file.
    """
    plt.figure(figsize=(10, 7))

    # Plot precision and recall vs threshold
    plt.plot(thresholds, precisions, label=f'{model_name} Precision', marker='o')
    plt.plot(thresholds, recalls, label=f'{model_name} Recall', marker='o')

    plt.xlabel('Thresholds')
    plt.ylabel(f'{ylabel1}/{ylabel2}')
    plt.title(title)
    plt.legend()
    plt.grid(True)

    # Save the plot
    plt.savefig(f'{base_dir}/{model_name.lower().replace(" ", "_")}_precision_recall_curve.png', format='png', dpi=600)
    plt.close()

def plot_and_save_f1_curve(thresholds, f1_scores_edpose, f1_scores_gesture, base_dir):
    """
    Plot the F1-score curve for both models and save it to a file.
    """
    plt.figure(figsize=(10, 7))

    # Plot F1-scores for ED-Pose and Gesture model
    plt.plot(thresholds, f1_scores_edpose, label='ED-Pose F1-Score', marker='o')
    plt.plot(thresholds, f1_scores_gesture, label='Gesture Model F1-Score', marker='o')

    plt.xlabel('Thresholds')
    plt.ylabel('F1-Score')
    plt.title('F1-Score Curve for ED-Pose and Gesture Model')
    plt.legend()
    plt.grid(True)

    # Save the plot
    plt.savefig(f'{base_dir}/f1_score_curve.png', format='png', dpi=600)
    plt.close()


# Call the confusion matrix plotting function after qualitative analysis
if __name__ == "__main__":
    # Load prediction files for both models
    pred_edpose_file = load_json(PRED_EDPOSE_FILE_PATH)
    pred_gesture_file = load_json(PRED_GESTURE_FILE_PATH)

    # Initialize COCO ground truth
    coco_gt = COCO(GT_FILE)

    # Set the IoU threshold for matching and the confidence threshold
    iou_threshold = 0.9  # Adjust IoU threshold as needed
    score_threshold = 0.15  # Adjust confidence threshold as needed

    # Get the list of class names from COCO dataset
    class_names = [cat['name'] for cat in coco_gt.loadCats(coco_gt.getCatIds())]

    # Perform qualitative analysis
    # perform_qualitative_analysis(coco_gt, pred_edpose_file, pred_gesture_file, BASE_DIR, threshold=score_threshold, iou_threshold=iou_threshold)
    thresholds = np.arange(0.01, 0.2, 0.01)
    print(thresholds)
    # iou_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    # for threshold in thresholds:
    #     for iou_threshold in iou_thresholds:
    #         plot_confusion_matrix(coco_gt, pred_edpose_file, pred_gesture_file, class_names, threshold=threshold, iou_threshold=iou_threshold)
    
    # Plot confusion matrix for both ED-Pose and our model
    plot_confusion_matrix(coco_gt, pred_edpose_file, pred_gesture_file, class_names, threshold=score_threshold, iou_threshold=iou_threshold)
    # generate_metrics_at_thresholds(coco_gt, pred_edpose_file, pred_gesture_file, class_names, thresholds, BASE_DIR)
    # plot_class_distribution(coco_gt, class_names)
    # print(class_names)