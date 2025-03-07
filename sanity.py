import os
import json

# Paths to COCO annotations
coco_dir = "../datasets/stratified_folds_unique_margin_8_n5/fold_0/annotations"
train_ann = os.path.join(coco_dir, "person_keypoints_train2017.json")
val_ann = os.path.join(coco_dir, "person_keypoints_val2017.json")
test_ann = os.path.join(coco_dir, "person_keypoints_test2017.json")  # If available

def get_filenames_and_categories(annotation_file):
    """Extracts filenames and annotation category counts from a COCO annotation file."""
    with open(annotation_file, 'r') as f:
        data = json.load(f)

    filenames = [img["file_name"] for img in data["images"]]
    category_counts = {}

    for ann in data["annotations"]:
        category_id = ann["category_id"]
        category_counts[category_id] = category_counts.get(category_id, 0) + 1
    
    return filenames, category_counts

# Load filenames and category counts for train, val, and test
train_filenames, train_category_counts = get_filenames_and_categories(train_ann)
val_filenames, val_category_counts = get_filenames_and_categories(val_ann)
test_filenames, test_category_counts = get_filenames_and_categories(test_ann) if os.path.exists(test_ann) else ([], {})

# Convert lists to sets for fast intersection check
train_set = set(train_filenames)
val_set = set(val_filenames)
test_set = set(test_filenames)

# Check for duplicate filenames
train_val_overlap = train_set & val_set
train_test_overlap = train_set & test_set

# Report results
if train_val_overlap:
    print(f"⚠️ WARNING: {len(train_val_overlap)} duplicate images found between TRAIN and VAL!")
    print(list(train_val_overlap)[:10])  # Print only first 10 duplicates

if train_test_overlap:
    print(f"⚠️ WARNING: {len(train_test_overlap)} duplicate images found between TRAIN and TEST!")
    print(list(train_test_overlap)[:10])  # Print only first 10 duplicates

if not train_val_overlap and not train_test_overlap:
    print("✅ Sanity check passed: No duplicate filenames between train, val, and test.")

# Print category-wise annotation counts
print("\n📊 Annotation Counts Per Category:")
print("🔹 Train Set:")
for category_id, count in sorted(train_category_counts.items()):
    print(f"   - Category {category_id}: {count} annotations")

print("\n🔹 Val Set:")
for category_id, count in sorted(val_category_counts.items()):
    print(f"   - Category {category_id}: {count} annotations")

if test_category_counts:
    print("\n🔹 Test Set:")
    for category_id, count in sorted(test_category_counts.items()):
        print(f"   - Category {category_id}: {count} annotations")
