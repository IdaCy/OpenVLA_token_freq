import os
import json
import shutil

# Paths
coco_images_dir = "data/raw-coco/train2014"
coco_annotations_file = "data/raw-coco/annotations/captions_train2014.json"
output_dir = "data/r10k-coco"
batch_size = 1000  # Move 1000 images per batch
log_file = os.path.join(output_dir, "moved_images.log")
failed_log_file = os.path.join(output_dir, "failed_moves.log")
captions_all_file = os.path.join(output_dir, "captions_all.json")  # Append all captions here

# Ensure output directories exist
os.makedirs(f"{output_dir}/train2014", exist_ok=True)

# Load COCO annotations
with open(coco_annotations_file, "r") as f:
    coco_data = json.load(f)

# Get remaining image filenames in the source directory
all_images = sorted([img for img in os.listdir(coco_images_dir) if img.endswith(".jpg")])

# Stop execution if fewer than batch_size images are left
if len(all_images) < batch_size:
    print(f"Only {len(all_images)} images left, stopping early.")
    exit()

# Select the first 1000 images (deterministic per run)
selected_images = set(all_images[:batch_size])

# **Fix: Create a mapping from filename -> image_id using COCO metadata**
filename_to_id = {img["file_name"]: img["id"] for img in coco_data["images"]}

# Ensure only images that exist in COCO annotations are selected
selected_image_ids = {filename_to_id[img]: img for img in selected_images if img in filename_to_id}

# **Warn if some images have no annotations**
missing_annotations = [img for img in selected_images if img not in filename_to_id]
if missing_annotations:
    print(f"Warning: {len(missing_annotations)} images have no annotations and will be skipped.")

# **Fix: Filter annotations by COCO metadata instead of extracted IDs**
filtered_annotations = {
    "images": [img for img in coco_data["images"] if img["id"] in selected_image_ids],
    "annotations": [ann for ann in coco_data["annotations"] if ann["image_id"] in selected_image_ids]
}

# Determine batch number for annotation saving
existing_batches = [f for f in os.listdir(output_dir) if f.startswith("captions_batch_")]
batch_number = len(existing_batches) + 1

# Save new annotation file per batch
captions_batch_file = os.path.join(output_dir, f"captions_batch_{batch_number}.json")
with open(captions_batch_file, "w") as f:
    json.dump(filtered_annotations, f)

# **Fix: Append captions to a master file instead of overwriting**
if os.path.exists(captions_all_file):
    with open(captions_all_file, "r") as f:
        all_captions = json.load(f)
else:
    all_captions = {"images": [], "annotations": []}

# Append new batch to the master captions file
all_captions["images"].extend(filtered_annotations["images"])
all_captions["annotations"].extend(filtered_annotations["annotations"])

# Save updated master captions file
with open(captions_all_file, "w") as f:
    json.dump(all_captions, f)

# Move images to new directory safely
moved_files = []
failed_files = []
try:
    for img_id, filename in selected_image_ids.items():
        src_path = os.path.join(coco_images_dir, filename)
        dst_path = os.path.join(output_dir, "train2014", filename)

        try:
            shutil.move(src_path, dst_path)  # Move file
            moved_files.append(filename)  # Log successful move
        except Exception as move_error:
            print(f"Error moving {filename}: {move_error}")
            failed_files.append(filename)  # Log failed moves

    # Log moved files for safety (appends to prevent overwriting)
    with open(log_file, "a") as f:
        for filename in moved_files:
            f.write(filename + "\n")

    # Log failed moves separately (appends)
    if failed_files:
        with open(failed_log_file, "a") as f:
            for filename in failed_files:
                f.write(filename + "\n")

    print(f"Successfully moved {len(moved_files)} images for batch {batch_number} and saved corresponding annotations.")

except Exception as e:
    print(f"Fatal error occurred: {e}")
    print("Some images may not have been moved. Check the log file for consistency.")
