import os
import torch
from transformers import AutoModel, AutoTokenizer
from datasets import load_from_disk
import numpy as np

# Paths
DATA_PATH = "data/raw-multw/"
OUTPUT_DIR = "results_multiwoz/"

# Model checkpoint
MODEL_NAME = "models/openvla/openvla-7b"

# Load MultiWOZ dataset
print("Loading MultiWOZ dataset...")
dataset = load_from_disk(DATA_PATH)

# Ensure output directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "activations"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "attention"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "output"), exist_ok=True)

# Load model and tokenizer
print("Loading model and tokenizer...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModel.from_pretrained(MODEL_NAME, output_attentions=True,
                                  output_hidden_states=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()


# Process dialogues
def process_dialogue(dialogue, index):
    inputs = tokenizer(dialogue, return_tensors="pt", padding=True,
                       truncation=True, max_length=512).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # Extract activations (hidden states), attention, and final output
    activations = outputs.hidden_states  # List of layers (tuple of tensors)
    attentions = outputs.attentions  # List of attention tensors per layer
    output_logits = outputs.last_hidden_state.cpu().numpy()

    # Save activations layer-wise
    for layer_idx, activation in enumerate(activations):
        layer_dir = os.path.join(OUTPUT_DIR, "activations",
                                 f"layer_{layer_idx}")
        os.makedirs(layer_dir, exist_ok=True)
        np.save(os.path.join(layer_dir, f"dialogue_{index}.npy"),
                activation.cpu().numpy())

    # Save attention values layer-wise
    for layer_idx, attention in enumerate(attentions):
        layer_dir = os.path.join(OUTPUT_DIR, "attention", f"layer_{layer_idx}")
        os.makedirs(layer_dir, exist_ok=True)
        np.save(os.path.join(layer_dir, f"dialogue_{index}.npy"),
                attention.cpu().numpy())

    # Save model output
    np.save(os.path.join(OUTPUT_DIR, "output", f"dialogue_{index}.npy"),
            output_logits)

    print(f"Processed dialogue {index+1}/{len(dataset['train'])}")


# Process all dialogues in the train split
print("Processing dialogues...")
for idx, dialogue in enumerate(dataset["train"]["turns"]):
    process_dialogue(" ".join(dialogue['utterance']), idx)

print("Processing complete. Results saved to:", OUTPUT_DIR)
