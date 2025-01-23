import os
from transformers import AutoModel, AutoTokenizer

model_name = "openvla/openvla-7b"  # Official model ID from Hugging Face
cache_dir = os.path.expanduser("~/OpenVLA-forget-tune/cache/")

try:
    print(f"Downloading model {model_name}...")

    # Load the model with trust_remote_code=True to allow loading custom code
    model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir,
                                      trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir,
                                              trust_remote_code=True)

    # Save model locally for future use
    save_dir = os.path.expanduser("~/OpenVLA-forget-tune/models/openvla-7b/")
    os.makedirs(save_dir, exist_ok=True)  # Ensure directory exists
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    print(f"Model downloaded and saved successfully at {save_dir}")

except Exception as e:
    print(f"An error occurred: {e}")
