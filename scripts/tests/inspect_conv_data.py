from datasets import load_from_disk

# Load datasets
coqa = load_from_disk("data/raw-coqa")
blskt = load_from_disk("data/raw-blskt")

# Print samples
print("CoQA Sample:", coqa["train"][0])
print("BlendedSkillTalk Sample:", blskt["train"][0])
