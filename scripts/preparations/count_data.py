import os


def get_size(start_path="."):
    """Quickly get folder size without deep recursion"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):  # Ignore broken symlinks
                total_size += os.path.getsize(fp)
    return total_size


# Define datasets
datasets = ["data/raw-vqa", "data/raw-coco", "data/raw-multw",
            "data/raw-bridge", "data/raw-natsgd"]

# Get sizes and print
for dataset in datasets:
    size = get_size(dataset) / (1024 ** 3)  # Convert bytes → GB
    print(f"{dataset}: {size:.2f} GB")
