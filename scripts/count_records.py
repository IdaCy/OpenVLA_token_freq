import tensorflow as tf
import os

data_dir = "data/raw-DROID/"

# Find all TFRecord files in the directory
tfrecord_files = [os.path.join(data_dir, f) for f in os.listdir(
    data_dir) if 'tfrecord' in f]

total_records = 0

for file_path in tfrecord_files:
    try:
        dataset = tf.data.TFRecordDataset(file_path)
        count = sum(1 for _ in dataset)
        print(f"Total records in {file_path}: {count}")
        total_records += count
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

print(f"Total number of records across all files: {total_records}")
