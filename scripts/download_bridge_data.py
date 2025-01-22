import requests
import zipfile
import os
import time

# Configuration
DATASET_URL = "https://rail.eecs.berkeley.edu/datasets/bridge_release/data/scripted_6_18.zip"
DATASET_PATH = "data/raw-bridge/scripted_6_18.zip"
TEMP_DOWNLOAD_PATH = "data/raw-bridge/scripted_6_18.tmp.zip"
EXTRACTED_PATH = "data/raw-bridge/extracted/"
TARGET_SEQUENCES = 10000
CHUNK_SIZE = 1024 * 1024  # 1MB chunk size
MAX_RETRIES = 3  # Retry download up to 3 times if it fails


def download_dataset():
    """Download the dataset incrementally and verify integrity."""
    if os.path.exists(DATASET_PATH) and os.path.getsize(DATASET_PATH) > 1024:
        print("Existing file found. Skipping download.")
        return

    print("Downloading dataset...")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(DATASET_URL, stream=True, timeout=60)
            response.raise_for_status()  # Check for HTTP errors

            total_size = int(response.headers.get('content-length', 0)) / (1024 * 1024)
            with open(TEMP_DOWNLOAD_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    f.write(chunk)
                    downloaded_size = os.path.getsize(TEMP_DOWNLOAD_PATH) / (1024 * 1024)
                    print(f"Downloaded: {downloaded_size:.2f} MB of {total_size:.2f} MB", end="\r")

            # Move to final path if download successful
            os.rename(TEMP_DOWNLOAD_PATH, DATASET_PATH)
            print("\nDownload complete.")

            # Validate the downloaded file size
            if os.path.getsize(DATASET_PATH) == 0:
                os.remove(DATASET_PATH)
                raise ValueError("Downloaded file is empty.")

            break  # Successful download, exit retry loop

        except requests.RequestException as e:
            print(f"\nAttempt {attempt} failed: {e}")
            if os.path.exists(TEMP_DOWNLOAD_PATH):
                os.remove(TEMP_DOWNLOAD_PATH)

            if attempt < MAX_RETRIES:
                print("Retrying in 10 seconds...")
                time.sleep(10)
            else:
                raise RuntimeError("Maximum download attempts reached. Exiting.")


def extract_subset():
    """Extract the required sequences from the ZIP dataset."""
    print("Extracting subset from dataset...")

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")

    with zipfile.ZipFile(DATASET_PATH, "r") as zip_ref:
        file_list = zip_ref.namelist()
        action_files = [f for f in file_list if "actions" in f and f.endswith(".hdf5")]

        if len(action_files) < TARGET_SEQUENCES:
            raise ValueError(f"ZIP archive contains only {len(action_files)} action files, "
                             f"but {TARGET_SEQUENCES} are needed.")

        os.makedirs(EXTRACTED_PATH, exist_ok=True)

        for i, file in enumerate(action_files[:TARGET_SEQUENCES]):
            zip_ref.extract(file, EXTRACTED_PATH)
            print(f"Extracted {i + 1}/{TARGET_SEQUENCES}: {file}")

        print(f"Successfully extracted {TARGET_SEQUENCES} action sequences to {EXTRACTED_PATH}")

    # Verify the number of extracted files
    extracted_count = len(os.listdir(EXTRACTED_PATH))
    if extracted_count != TARGET_SEQUENCES:
        raise RuntimeError(f"Expected {TARGET_SEQUENCES} files, but found {extracted_count}")


if __name__ == "__main__":
    os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)

    download_dataset()
    extract_subset()

    print("Process completed successfully.")
