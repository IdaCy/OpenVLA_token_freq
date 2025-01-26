from datasets import load_from_disk

# Load the dataset
dataset = load_from_disk("~/OpenVLA-forget-tune/data/raw-multw/")

# Check the splits
print(dataset)

# Inspect a sample dialogue
print(dataset["train"][0])
