import os
import subprocess
import tensorflow as tf

# Constants
BUCKET_PATH = "gs://gresearch/robotics/droid/1.0.0/"
DATA_DIR = "data/raw-DROID/"
DESIRED_RECORD_COUNT = 100000
BATCH_SIZE = 50  # Number of files to download in each batch
START_INDEX = 0  # Start from file 00000
TOTAL_FILES = 2048  # Total available files in dataset


def count_tfrecord_records(file_path):
    """Count the number of records in a TFRecord file."""
    try:
        dataset = tf.data.TFRecordDataset(file_path)
        count = sum(1 for _ in dataset)
        return count
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return 0


def total_records_in_directory(directory):
    """Count all records in the downloaded TFRecord files."""
    total_records = 0
    tfrecord_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.tfrecord')]
    
    for tfrecord_file in tfrecord_files:
        record_count = count_tfrecord_records(tfrecord_file)
        print(f"{tfrecord_file}: {record_count} records")
        total_records += record_count

    print(f"Total number of records across all files: {total_records}")
    return total_records


def download_tfrecords(start, batch_size):
    """Download a batch of TFRecord files."""
    file_list = [
        f"{BUCKET_PATH}r2d2_faceblur-train.tfrecord-{str(i).zfill(5)}-of-02048"
        for i in range(start, start + batch_size)
    ]
    
    print(f"Downloading files {start} to {start + batch_size - 1}...")
    cmd = ["gsutil", "-m", "cp"] + file_list + [DATA_DIR]
    
    try:
        subprocess.run(cmd, check=True)
        print("Download complete.")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading files: {e}")


def main():
    """Main function to download files until desired record count reached."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    current_record_count = total_records_in_directory(DATA_DIR)
    file_index = START_INDEX

    while current_record_count < DESIRED_RECORD_COUNT and file_index < TOTAL_FILES:
        download_tfrecords(file_index, BATCH_SIZE)
        current_record_count = total_records_in_directory(DATA_DIR)
        file_index += BATCH_SIZE

        if current_record_count >= DESIRED_RECORD_COUNT:
            print(f"Reached desired record count: {current_record_count}")
            break
        elif file_index >= TOTAL_FILES:
            print("Reached maximum available files.")
            break

    print(f"Final record count: {current_record_count}")
    print("Dataset download complete.")


if __name__ == "__main__":
    main()
