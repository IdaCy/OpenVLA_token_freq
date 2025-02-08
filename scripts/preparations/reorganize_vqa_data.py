import os
import shutil

# Define main directory
BASE_DIR = "stage1_inf_vqa"

# Ensure subdirectories exist
subdirs = ["activations", "attentions", "logits", "probabilities", "representations"]
for sub in subdirs:
    os.makedirs(os.path.join(BASE_DIR, sub), exist_ok=True)

# Get all files in the base folder (not in subdirectories yet)
files = [f for f in os.listdir(BASE_DIR) if os.path.isfile(os.path.join(BASE_DIR, f)) and f.endswith(".pkl")]

# Move files to the correct directories
for file in files:
    file_path = os.path.join(BASE_DIR, file)

    if "_activations.pkl" in file:
        shutil.move(file_path, os.path.join(BASE_DIR, "activations", file))
    elif "_attentions.pkl" in file:
        shutil.move(file_path, os.path.join(BASE_DIR, "attentions", file))
    elif "_logits.pkl" in file:
        shutil.move(file_path, os.path.join(BASE_DIR, "logits", file))
    elif "_probabilities.pkl" in file:
        shutil.move(file_path, os.path.join(BASE_DIR, "probabilities", file))
    elif "_representations.pkl" in file:
        shutil.move(file_path, os.path.join(BASE_DIR, "representations", file))
    else:
        print(f"⚠️ Warning: File {file} does not match expected patterns and was not moved.")

print("✅ Reorganization complete! Files are now structured properly.")
