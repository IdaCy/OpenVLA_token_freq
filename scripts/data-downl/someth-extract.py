import tarfile
import os

# Define paths
tar_path = "data/raw-someth/20bn-something-something-v2.tar.gz"
extract_path = "data/raw-someth/20bn-something-something-v2/"

# Ensure the target directory exists
os.makedirs(extract_path, exist_ok=True)

# Extract with progress updates
with tarfile.open(tar_path, "r:gz") as tar:
    members = tar.getmembers()
    total_files = len(members)

    print(f"Extracting {total_files} files...\n")

    for i, member in enumerate(members, 1):
        tar.extract(member, path=extract_path)
        if i % 4000 == 0:  # Print progress every 4000 files
            print(f"Extracted {i}/{total_files} files...")

print("✅ Extraction complete!")
