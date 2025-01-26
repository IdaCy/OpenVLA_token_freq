from datasets import load_dataset

dataset_path = "data/raw-multw"

# Explicitly tell Hugging Face to use the "arrow" format
train_dataset = load_dataset("arrow", data_files={"train": f"{dataset_path}/train/data-00000-of-00001.arrow"})
val_dataset = load_dataset("arrow", data_files={"validation": f"{dataset_path}/validation/data-00000-of-00001.arrow"})
test_dataset = load_dataset("arrow", data_files={"test": f"{dataset_path}/test/data-00000-of-00001.arrow"})

# Print counts to verify
print(f"Train split size: {len(train_dataset['train'])}")
print(f"Validation split size: {len(val_dataset['validation'])}")
print(f"Test split size: {len(test_dataset['test'])}")

# Print a sample for inspection
print("Sample train entry:", train_dataset["train"][0])
print("Sample validation entry:", val_dataset["validation"][0])
print("Sample test entry:", test_dataset["test"][0])
