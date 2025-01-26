import os
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor

# Define the model and save paths
model_name = "openvla/openvla-7b"
cache_dir = os.path.expanduser("~/OpenVLA-forget-tune/cache/")
save_dir = os.path.expanduser("~/OpenVLA-forget-tune/models/openvla-7b/")

try:
    print(f"Downloading model {model_name}...")

    # Load the model and processor
    processor = AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir,
                                              trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        attn_implementation="flash_attention_2", 
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )

    # Ensure directories exist and save locally
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)

    print(f"Model and processor saved successfully at {save_dir}")

except Exception as e:
    print(f"An error occurred: {e}")
