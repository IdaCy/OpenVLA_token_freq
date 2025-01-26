from datasets import Dataset

# Load the dataset manually from the .arrow file
train_dataset = Dataset.from_file("data/raw-multw/train/data-00000-of-00001.arrow")

# Check dataset size and sample contents
print(f"Total train examples: {len(train_dataset)}")
print(train_dataset[0])  # Print a sample data entry
