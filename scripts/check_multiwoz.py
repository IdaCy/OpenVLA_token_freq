from datasets import load_from_disk

dataset_path = "~/OpenVLA-forget-tune/data/raw-multw/"
dataset = load_from_disk(dataset_path)

print(dataset)  # Check if train/val/test splits are loaded properly

print("Train samples:", len(dataset["train"]))
print("Validation samples:", len(dataset["validation"]))
print("Test samples:", len(dataset["test"]))

print(dataset["train"][0])  # Print first dialogue to verify contents
