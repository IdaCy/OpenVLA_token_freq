import os
import json
import cv2
import numpy as np
import tensorflow as tf
import pickle  # For robust serialization
from rlds import rlds_types
import logging
import gc  # For explicit garbage collection

# **SET UP LOGGING**
logging.basicConfig(
    filename="logs/prep_someth8G_debug.log",
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.DEBUG
)

# **Paths** – using the 8G subset paths
VIDEO_DIR = "data/someth_8G/videos/"
TRAIN_FILE = "data/someth_8G/labels/train.json"
VAL_FILE = "data/someth_8G/labels/validation.json"
LABELS_FILE = "data/someth_8G/labels/labels.json"
OUTPUT_TFRECORD = "datasets/open-x-embodiment/somethv2/somethv2_rlds.tfrecord"

# **Configuration for Compression/Downscaling**
# Set the target resolution. For example, (160, 120) reduces the number of pixels by 4x compared to 320x240.
RESIZE_DIM = (160, 120)  
# Set sample_rate to > 1 if you want to keep only every nth frame (1 = every frame)
SAMPLE_RATE = 1  
# JPEG quality from 0 to 100 (higher means better quality and larger size)
JPEG_QUALITY = 90

# **Load General Labels (`labels.json`)**
try:
    with open(LABELS_FILE, "r") as f:
        action_labels = json.load(f)  # Maps generic action descriptions to IDs
    logging.info(f"Loaded {len(action_labels)} general action labels")
except Exception as e:
    logging.error(f"Failed to load action labels: {e}")
    raise

# **Preload existing video files**
existing_videos = set(os.listdir(VIDEO_DIR))
logging.info(f"Found {len(existing_videos)} available video files.")

# **Load Training and Validation Annotations**
def load_annotations(json_file):
    try:
        with open(json_file, "r") as f:
            data = json.load(f)
        logging.info(f"Loaded {len(data)} annotations from {json_file}")
        return data
    except Exception as e:
        logging.error(f"Failed to load annotations from {json_file}: {e}")
        return []

train_annotations = load_annotations(TRAIN_FILE)
val_annotations = load_annotations(VAL_FILE)
annotations = train_annotations + val_annotations
logging.info(f"Total annotations combined: {len(annotations)}")

MISSING_LABELS = set()
next_action_id = len(action_labels)  # numbering new labels after the existing

# **Find Closest Matching Label or Generate a New One**
def get_action_id(annotation_label):
    """
    Try to match a specific action label to a general category.
    If no match exists, assign a new action ID dynamically.
    """
    global action_labels, next_action_id

    # Look for an exact match first
    if annotation_label in action_labels:
        return int(action_labels[annotation_label])

    # Check if any canonical key appears in the annotation label (case-insensitive)
    for general_label in action_labels.keys():
        if general_label.lower() in annotation_label.lower():
            return int(action_labels[general_label])

    # If no match found, generate a new ID
    if annotation_label not in action_labels:
        action_labels[annotation_label] = str(next_action_id)
        next_action_id += 1
        MISSING_LABELS.add(annotation_label)
        logging.info(f"Generated new action ID {next_action_id - 1} for '{annotation_label}'")

    return int(action_labels[annotation_label])

# **Convert Video to Compressed Frames**
def video_to_frames(video_path, resize_dim=RESIZE_DIM, sample_rate=SAMPLE_RATE):
    """
    Extract frames from a video file, optionally downscale them and encode as JPEG.
    Returns a list of JPEG-encoded byte strings.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    if not cap.isOpened():
        logging.warning(f"Cannot open video: {video_path}")
        return frames  # Skip if the video cannot be opened

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Sample only every nth frame
        if frame_count % sample_rate != 0:
            frame_count += 1
            continue
        # Optionally resize frame
        if resize_dim is not None:
            frame = cv2.resize(frame, resize_dim)
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Encode frame as JPEG
        ret2, buf = cv2.imencode('.jpg', frame_rgb, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if ret2:
            frames.append(buf.tobytes())
        frame_count += 1

    cap.release()
    return frames

# **Create RLDS Episode**
def create_episode(annotation):
    """
    Convert an annotation into an RLDS episode.
    Each episode will now contain compressed JPEG images.
    """
    video_id = annotation.get("id", None)
    if not video_id or f"{video_id}.webm" not in existing_videos:
        logging.warning(f"Video {video_id} not found in {VIDEO_DIR}. Skipping.")
        return None  # Skip missing videos

    video_path = os.path.join(VIDEO_DIR, f"{video_id}.webm")
    frames = video_to_frames(video_path)
    if not frames:
        logging.warning(f"No frames extracted for video {video_path}.")
        return None

    # Map label to action ID using the free-form "label" field.
    action_str = annotation.get("label", None)
    action = get_action_id(action_str)

    steps = []
    for frame_bytes in frames:
        # Instead of storing a raw numpy array, store the JPEG-compressed bytes.
        steps.append({
            rlds_types.OBSERVATION: {"image": frame_bytes},
            rlds_types.ACTION: action,
            rlds_types.IS_TERMINAL: False
        })

    if steps:
        steps[-1][rlds_types.IS_TERMINAL] = True  # Mark last frame as terminal

    return {
        rlds_types.STEPS: steps,
        "metadata": {
            "instruction": annotation.get("template", ""),
            "placeholders": annotation.get("placeholders", [])
        }
    }

# **Write Episodes to TFRecord using pickle for serialization**
def write_episode_to_tfrecord(episode, writer):
    """Serialize and write a single episode to TFRecord."""
    serialized_episode = pickle.dumps(episode)
    example = tf.train.Example(features=tf.train.Features(feature={
        "episode": tf.train.Feature(bytes_list=tf.train.BytesList(value=[serialized_episode]))
    }))
    writer.write(example.SerializeToString())

# **Main Function: Streaming Processing to Save Memory**
def main():
    logging.info("Starting RLDS dataset creation...")
    total_annotations = len(annotations)
    valid_count = 0

    with tf.io.TFRecordWriter(OUTPUT_TFRECORD) as writer:
        for i, annotation in enumerate(annotations):
            if i % 5000 == 0:
                logging.info(f"Processing annotation {i+1}/{total_annotations}")
            episode = create_episode(annotation)
            # Write episode immediately if valid.
            if episode:
                valid_count += 1
                write_episode_to_tfrecord(episode, writer)
            # Explicitly delete the episode (and collect garbage) to free memory.
            del episode
            if i % 5000 == 0:
                gc.collect()

    logging.info(f"Finished processing. Total annotations: {total_annotations}. Valid episodes: {valid_count}")
    if MISSING_LABELS:
        logging.warning(f"{len(MISSING_LABELS)} new labels were added! Examples: {list(MISSING_LABELS)[:10]}")

if __name__ == "__main__":
    main()
