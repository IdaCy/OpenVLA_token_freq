import zipfile
import os
import json

# Paths
ZIP_PATH = "data/raw-vqa/train2014.zip"
TARGET_FOLDER = "data/raw-vqa/train2014"
QUESTION_FILE = "data/raw-vqa/v2_OpenEnded_mscoco_train2014_questions.json"

# Ensure target directory exists
os.makedirs(TARGET_FOLDER, exist_ok=True)

# Load VQA question file
with open(QUESTION_FILE, "r") as f:
    vqa_questions = json.load(f)["questions"]

print("loaded question file")

# Get unique image IDs from questions
required_image_ids = {q["image_id"] for q in vqa_questions}
print(f"Found {len(required_image_ids)} unique VQA image IDs.")

# Map image IDs to COCO filenames
required_filenames = {
    f"train2014/COCO_train2014_{img_id:012d}.jpg" for img_id in required_image_ids}

# Extract only necessary images
with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
    extracted_count = 0
    for filename in required_filenames:
        try:
            zip_ref.extract(filename, TARGET_FOLDER)
            extracted_count += 1
            if extracted_count % 500 == 0:
                print(f"Extracted {extracted_count}/ " +
                      f"{len(required_filenames)} images...")
        except KeyError:
            print(f"WARNING: Missing {filename} in ZIP!")

print(f"Extraction complete! Extracted {extracted_count} images for VQA V2.")
