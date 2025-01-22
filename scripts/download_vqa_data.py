import requests
import zipfile
import os
import json
import time

# Configuration
VQA_URLS = {
    "images_train": "http://images.cocodataset.org/zips/train2014.zip",
    "questions_train": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip",
    "annotations_train": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip"
}

DOWNLOAD_DIR = "data/raw-vqa"
REQUIRED_IMAGE_COUNT = 10000
CHUNK_SIZE = 1024 * 1024  # 1MB chunk size
MAX_RETRIES = 3  # Retry download up to 3 times if it fails


def download_file(url, output_path):
    """Download a file with retries."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if os.path.exists(output_path) and os.path.getsize(output_path) > 1024:
                print(f"File {output_path} already exists and is valid.")
                return
            print(f"Downloading {url} (Attempt {attempt}/{MAX_RETRIES})...")
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()

            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    f.write(chunk)
                    downloaded_size = os.path.getsize(output_path) / (1024 * 1024)
                    print(f"Downloaded: {downloaded_size:.2f} MB", end="\r")

            print(f"\nDownloaded successfully: {output_path}")
            return

        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt} failed: {e}")
            if attempt == MAX_RETRIES:
                raise RuntimeError(f"Failed to download {url}")
            print("Retrying in 10 seconds...")
            time.sleep(10)


def extract_subset(zip_path, target_folder, subset_count):
    """Extract a subset of images from the ZIP archive."""
    print(f"Extracting {subset_count} images from {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        image_files = [f for f in zip_ref.namelist() if f.endswith(".jpg")]
        
        if len(image_files) < subset_count:
            raise ValueError(f"ZIP archive contains only {len(image_files)} images, "
                             f"but {subset_count} were requested.")

        os.makedirs(target_folder, exist_ok=True)
        for i, file in enumerate(image_files[:subset_count]):
            zip_ref.extract(file, target_folder)
            print(f"Extracted {i + 1}/{subset_count}: {file}")

    print(f"Extracted {subset_count} images to {target_folder}")


def extract_questions_and_answers(question_zip, answer_zip, subset_count):
    """Extract and save a subset of questions and answers."""
    with zipfile.ZipFile(question_zip, 'r') as q_zip, zipfile.ZipFile(answer_zip, 'r') as a_zip:
        q_zip.extractall(DOWNLOAD_DIR)
        a_zip.extractall(DOWNLOAD_DIR)

    questions_file = os.path.join(DOWNLOAD_DIR, "v2_OpenEnded_mscoco_train2014_questions.json")
    answers_file = os.path.join(DOWNLOAD_DIR, "v2_mscoco_train2014_annotations.json")

    with open(questions_file, 'r') as f:
        questions = json.load(f)["questions"][:subset_count]
    with open(answers_file, 'r') as f:
        answers = json.load(f)["annotations"][:subset_count]

    with open(os.path.join(DOWNLOAD_DIR, "subset_questions.json"), 'w') as f:
        json.dump(questions, f)
    with open(os.path.join(DOWNLOAD_DIR, "subset_answers.json"), 'w') as f:
        json.dump(answers, f)

    print(f"Saved {subset_count} questions and answers.")


def main():
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    # Download necessary files
    download_file(VQA_URLS["images_train"], os.path.join(DOWNLOAD_DIR, "train2014.zip"))
    download_file(VQA_URLS["questions_train"], os.path.join(DOWNLOAD_DIR, "vqa_questions_train.zip"))
    download_file(VQA_URLS["annotations_train"], os.path.join(DOWNLOAD_DIR, "vqa_annotations_train.zip"))

    # Extract images and questions/answers
    extract_subset(os.path.join(DOWNLOAD_DIR, "train2014.zip"), os.path.join(DOWNLOAD_DIR, "train2014"), REQUIRED_IMAGE_COUNT)
    extract_questions_and_answers(
        os.path.join(DOWNLOAD_DIR, "vqa_questions_train.zip"),
        os.path.join(DOWNLOAD_DIR, "vqa_annotations_train.zip"),
        REQUIRED_IMAGE_COUNT
    )

    print("VQA dataset download and extraction complete.")


if __name__ == "__main__":
    main()
