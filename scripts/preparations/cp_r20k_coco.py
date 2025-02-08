import os
import json
import shutil

# Paths
coco_images_dir = "data/raw-coco/train2014"
coco_annotations_file = "data/raw-coco/annotations/captions_train2014.json"
output_dir = "data/r10k-3_coco"
log_file = os.path.join(output_dir, "moved_images.log")
failed_log_file = os.path.join(output_dir, "failed_moves.log")
captions_all_file = os.path.join(output_dir, "captions_all.json")

# Ensure output directories exist
os.makedirs(f"{output_dir}/train2014", exist_ok=True)

# Load COCO annotations
with open(coco_annotations_file, "r") as f:
    coco_data = json.load(f)

# Get all remaining image filenames in the source directory
all_images = sorted([img for img in os.listdir(coco_images_dir) if img.endswith(".jpg")])

# Stop execution if fewer than 10,000 images are left
if len(all_images) < 10000:
    print(f"⚠ Only {len(all_images)} images left, stopping early.")
    exit()

# Select the first 10,000 images
selected_images = set(all_images[:10000])

# **Create a mapping from filename -> image_id using COCO metadata**
filename_to_id = {img["file_name"]: img["id"] for img in coco_data["images"]}

# **Ensure every selected image has a caption**
selected_image_ids = {
    filename_to_id[img]: img for img in selected_images if img in filename_to_id
}

# **Warn if some images have no annotations**
missing_annotations = selected_images - set(filename_to_id.keys())
if missing_annotations:
    print(f"⚠ WARNING: {len(missing_annotations)} images have no captions! Skipping them.")
    with open("missing_captions.log", "w") as f:
        f.writelines(f"{img}\n" for img in missing_annotations)

# **Filter annotations for the selected images**
filtered_annotations = {
    "images": [img for img in coco_data["images"] if img["id"] in selected_image_ids],
    "annotations": [ann for ann in coco_data["annotations"] if ann["image_id"] in selected_image_ids]
}

# **Save the annotations file**
with open(captions_all_file, "w") as f:
    json.dump(filtered_annotations, f)

# **Move images to the new directory safely**
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
            print(f"⚠ Error moving {filename}: {move_error}")
            failed_files.append(filename)  # Log failed moves

    # **Log moved images for safety (append mode)**
    with open(log_file, "a") as f:
        for filename in moved_files:
            f.write(filename + "\n")

    # **Log failed moves separately**
    if failed_files:
        with open(failed_log_file, "a") as f:
            for filename in failed_files:
                f.write(filename + "\n")
        print(f"⚠ WARNING: {len(failed_files)} images failed to move! Check '{failed_log_file}'.")

    print(f"✅ Successfully moved {len(moved_files)} images and saved their captions.")

except Exception as e:
    print(f"❌ Fatal error: {e}")
    print("Some images may not have been moved. Check the log files for details.")
