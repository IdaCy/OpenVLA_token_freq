import os
import json
import shutil

# Define source and destination paths
SOURCE_DIR = "data/coco-10k-1"
DEST_DIR = "data/raw-coco1k"
IMAGE_SRC = os.path.join(SOURCE_DIR, "train2014")
IMAGE_DEST = os.path.join(DEST_DIR, "train2014")
CAPTION_SRC = os.path.join(SOURCE_DIR, "captions_1.json")
CAPTION_DEST = os.path.join(DEST_DIR, "captions_1.json")

# Ensure destination directories exist
os.makedirs(IMAGE_DEST, exist_ok=True)

# Get a sorted list of image files
image_files = sorted([f for f in os.listdir(IMAGE_SRC) if f.endswith(".jpg")])
selected_images = image_files[:1000]  # First 1000 images

# Copy images
print(f"Copying {len(selected_images)} images...")
for img in selected_images:
    shutil.copy2(os.path.join(IMAGE_SRC, img), os.path.join(IMAGE_DEST, img))

print("✅ Images copied successfully!")

# Load and filter captions
if os.path.exists(CAPTION_SRC):
    with open(CAPTION_SRC, "r") as f:
        captions_data = json.load(f)

    # Filter captions for the selected images
    image_ids = {img.split("_")[-1].split(".")[0] for img in selected_images}
    filtered_captions = [cap for cap in captions_data["annotations"] if str(cap["image_id"]).zfill(12) in image_ids]

    # Save new captions JSON
    captions_data["annotations"] = filtered_captions
    with open(CAPTION_DEST, "w") as f:
        json.dump(captions_data, f, indent=4)

    print("✅ Captions copied successfully!")
else:
    print("⚠️ Warning: Captions file not found!")

print("🎉 Copying complete!")
