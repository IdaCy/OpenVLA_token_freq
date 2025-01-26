import os
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor

model_dir = os.path.expanduser("~/OpenVLA-forget-tune/models/openvla-7b/")

try:
    processor = AutoProcessor.from_pretrained(model_dir,
                                              trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(model_dir,
                                                   torch_dtype=torch.bfloat16,
                                                   trust_remote_code=True)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
