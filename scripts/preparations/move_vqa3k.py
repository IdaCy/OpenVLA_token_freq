import os
import json
import random
import shutil

# Source directories
SRC_VQA_DIR = "data/raw-vqa"
SRC_IMG_DIR = os.path.join(SRC_VQA_DIR, "train2014")
SRC_Q_FILE = os.path.join(SRC_VQA_DIR, "v2_OpenEnded_mscoco_train2014_questions.json")
SRC_A_FILE = os.path.join(SRC_VQA_DIR, "v2_mscoco_train2014_annotations.json")

# Destination directories
DEST_VQA_DIR = "data/raw-vqa3k"
DEST_IMG_DIR = os.path.join(DEST_VQA_DIR, "train2014")
DEST_Q_FILE = os.path.join(DEST_VQA_DIR, "v2_OpenEnded_mscoco_train2014_questions.json")
DEST_A_FILE = os.path.join(DEST_VQA_DIR, "v2_mscoco_train2014_annotations.json")

# Ensure destination directories exist
os.makedirs(DEST_VQA_DIR, exist_ok=True)
os.makedirs(DEST_IMG_DIR, exist_ok=True)

# Load full question and answer datasets
print("Loading VQA questions and answers...")
with open(SRC_Q_FILE, "r") as f:
    vqa_questions = json.load(f)

with open(SRC_A_FILE, "r") as f:
    vqa_answers = json.load(f)

# Extract all available image IDs from the dataset
all_img_ids = list(set(q["image_id"] for q in vqa_questions["questions"]))
random.shuffle(all_img_ids)  # Shuffle for randomness

# Select the first 3000 unique image IDs
selected_img_ids = set(all_img_ids[:3000])
print(f"Selected {len(selected_img_ids)} images for the VQA 3K subset.")

# Filter questions and answers related to the selected images
filtered_questions = [q for q in vqa_questions["questions"] if q["image_id"] in selected_img_ids]
filtered_answers = [a for a in vqa_answers["annotations"] if a["image_id"] in selected_img_ids]

print(f"Filtered {len(filtered_questions)} questions and {len(filtered_answers)} answers.")

# Save filtered questions and answers
print("Saving filtered questions and answers...")
with open(DEST_Q_FILE, "w") as f:
    json.dump({"questions": filtered_questions}, f, indent=4)

with open(DEST_A_FILE, "w") as f:
    json.dump({"annotations": filtered_answers}, f, indent=4)

# Copy images
print("Copying images...")
for img_id in selected_img_ids:
    img_filename = f"COCO_train2014_{img_id:012d}.jpg"
    src_img_path = os.path.join(SRC_IMG_DIR, img_filename)
    dest_img_path = os.path.join(DEST_IMG_DIR, img_filename)

    if os.path.exists(src_img_path):
        shutil.copy2(src_img_path, dest_img_path)  # Preserve metadata
    else:
        print(f"⚠️ WARNING: Missing image {img_filename}, skipping...")

print("✅ VQA 3K subset successfully created in data/raw-vqa3k/")
