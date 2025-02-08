from datasets import load_dataset

# Define the save directory
save_path = "data/raw-someth"

# Load dataset with remote code execution allowed
dataset = load_dataset("HuggingFaceM4/something_something_v2",
                       cache_dir=save_path,
                       trust_remote_code=True)

print(f"Dataset downloaded to: {save_path}")
