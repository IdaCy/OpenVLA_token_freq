import os
from huggingface_hub import HfApi, hf_hub_download

# Define the model repository name
model_name = "openvla/openvla-7b"

# Initialize the Hugging Face API
api = HfApi()

# List files in the repository
files = api.list_repo_files(repo_id=model_name, repo_type="model")
print("Files in the Hugging Face repository:")
for file in files:
    print(file)

# Files to check sizes
files_to_check = [
    "model-00001-of-00003.safetensors",
    "model-00002-of-00003.safetensors",
    "model-00003-of-00003.safetensors",
]

print("\nChecking file sizes...")
for file_name in files_to_check:
    try:
        file_path = hf_hub_download(repo_id=model_name, filename=file_name, repo_type="model", cache_dir="./", force_download=False)
        file_size = os.path.getsize(file_path) / (1024 ** 3)  # Convert to GB
        print(f"{file_name}: {file_size:.2f} GB")
    except Exception as e:
        print(f"Error checking {file_name}: {e}")
