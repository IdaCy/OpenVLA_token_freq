import json
import random
import os
import time
import subprocess

# ====================
# === Configuration ===
# ====================
# Adjust these paths if needed. For example, if your original data is in "data/orig_someth", change accordingly.
original_data_path = "data/orig_someth/20bn-something-something-v2/"
labels_path = "data/orig_someth/labels/"
output_data_path = "data/someth_8G/videos/"
output_labels_path = "data/someth_8G/labels/"
log_dir = "logs/"
copy_list_file = os.path.join(log_dir, "copy_list.txt")
log_file = os.path.join(log_dir, "copy_progress.log")
failed_log = os.path.join(log_dir, "failed_copies.log")

# Create necessary directories
os.makedirs(output_data_path, exist_ok=True)
os.makedirs(output_labels_path, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# ===============================
# === Load and Combine Metadata ===
# ===============================
json_files = ["train.json", "validation.json", "test.json"]
video_data = []

print("🔹 Loading JSON files...")
start_time = time.time()
for file in json_files:
    file_path = os.path.join(labels_path, file)
    with open(file_path, "r") as f:
        video_data.extend(json.load(f))
print(f"✅ Finished loading JSON files in {time.time() - start_time:.2f} seconds.")

# Build a dictionary mapping video id to its metadata entry (from the files where label exists)
video_dict = {entry["id"]: entry for entry in video_data if entry.get("label") is not None}

# ======================================
# === Group videos by their label text ===
# ======================================
label_counts = {}
for entry in video_dict.values():
    label = entry["label"]
    label_counts.setdefault(label, []).append(entry["id"])

# ======================================
# === Sampling rules and selection ===
# ======================================
selected_samples = []
total_target_samples = 80000  # target number of videos (adjust as needed for ~8GB)
max_per_class = 1600
min_per_class = 100

print("🔹 Selecting balanced subset...")
for label, video_ids in label_counts.items():
    count = len(video_ids)
    if count > 5000:
        sample_size = min(max_per_class, count)
    elif 500 < count <= 5000:
        sample_size = min(int(count * 0.18), count)
    elif 100 < count <= 500:
        sample_size = min(150, count)
    else:
        sample_size = min(max(min_per_class, count), count)
    selected_samples.extend(random.sample(video_ids, sample_size))

# Ensure we do not oversample
if len(selected_samples) > total_target_samples:
    selected_samples = random.sample(selected_samples, total_target_samples)

print(f"✅ Selected {len(selected_samples)} samples.")

# ======================================
# === Verify file existence before copying ===
# ======================================
valid_samples = []
for video_id in selected_samples:
    src = os.path.join(original_data_path, f"{video_id}.webm")
    if os.path.exists(src):
        valid_samples.append(video_id)
    else:
        print(f"⚠️ Warning: Missing file {src}")

selected_samples = valid_samples

# ======================================
# === Write the copy list file ===
# ======================================
with open(copy_list_file, "w") as f:
    for video_id in selected_samples:
        src = os.path.join(original_data_path, f"{video_id}.webm")
        dst = os.path.join(output_data_path, f"{video_id}.webm")
        # Write the source and destination separated by a space
        f.write(f"{src} {dst}\n")
print(f"✅ Batch copy list written to {copy_list_file}.")

# Log the start of the copying process
with open(log_file, "w") as log:
    log.write("Starting batch copy process...\n")

# ======================================
# === Copy files in parallel using xargs ===
# ======================================
print("🔹 Starting parallel copying...")
start_copy_time = time.time()
# Use xargs with -n 2 so that each line is split into two arguments (source and destination).
copy_cmd = (
    f"xargs -a {copy_list_file} -n 2 -P 8 bash -c 'cp \"$1\" \"$2\" || echo Failed: \"$1 $2\" >> {failed_log}' _"
)
subprocess.run(copy_cmd, shell=True, stderr=subprocess.STDOUT)
print(f"✅ Finished copying files in {time.time() - start_copy_time:.2f} seconds.")

# ======================================
# === Verify copied files ===
# ======================================
copied_files = set(os.listdir(output_data_path))
missing_files = [vid for vid in selected_samples if f"{vid}.webm" not in copied_files]
if missing_files:
    print(f"⚠️ Warning: {len(missing_files)} files were not copied. Check {failed_log}.")

# ======================================
# === Filter and Save Metadata ===
# ======================================
# For each metadata file, filter entries to those in selected_samples
for file in json_files:
    with open(os.path.join(labels_path, file), "r") as f:
        full_data = json.load(f)
    filtered_entries = [entry for entry in full_data if entry["id"] in selected_samples]
    out_path = os.path.join(output_labels_path, file)
    with open(out_path, "w") as f:
        json.dump(filtered_entries, f, indent=4)
    print(f"✅ Filtered metadata saved to {out_path}.")

# Copy labels.json unchanged
subprocess.run(f"cp {os.path.join(labels_path, 'labels.json')} {os.path.join(output_labels_path, 'labels.json')}", shell=True)
print(f"✅ Successfully created a balanced subset of {len(selected_samples)} videos and their metadata.")
