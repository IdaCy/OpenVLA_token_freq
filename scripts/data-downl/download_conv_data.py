import os
import time
from datasets import load_dataset

# Define target directories
coqa_dir = 'data/raw-coqa'
blskt_dir = 'data/raw-blskt'

# Create directories if they don't exist
os.makedirs(coqa_dir, exist_ok=True)
os.makedirs(blskt_dir, exist_ok=True)


# Function to retry downloading dataset
def download_with_retries(dataset_name, save_dir, max_retries=5, wait_time=30):
    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1}/{max_retries}: Downloading {dataset_name}...")
            dataset = load_dataset(dataset_name)
            dataset.save_to_disk(save_dir)
            print(f"✅ Successfully downloaded {dataset_name} to {save_dir}")
            return
        except Exception as e:
            print(f"⚠️ Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"❌ Failed to download {dataset_name} after {max_retries} attempts.")


# Download datasets with retries
download_with_retries('stanfordnlp/coqa', coqa_dir)
download_with_retries('ParlAI/blended_skill_talk', blskt_dir)

print("✅ All downloads completed.")
