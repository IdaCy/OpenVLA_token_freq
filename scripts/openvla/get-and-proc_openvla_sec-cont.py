import os
import sys
import timm
import numpy as np
from transformers import AutoModelForVision2Seq, AutoProcessor

import torch

# Set environment variables to writable locations
os.environ["HF_HOME"] = os.path.expanduser("~/OpenVLA-forget-tune/cache")
os.environ["TRANSFORMERS_CACHE"] = os.environ["HF_HOME"]

# Ensure cache directory exists
os.makedirs(os.environ["HF_HOME"], exist_ok=True)

# Fix for protobuf compatibility issue
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

print(f"✅ Using cache directory: {os.environ['HF_HOME']}")

# Check package versions
print("Using timm version:", timm.__version__)
print("NumPy version:", np.__version__, "from", np.__file__)

# Check if accelerate is available
try:
    from accelerate import infer_auto_device_map
except ImportError:
    print("❌ The 'accelerate' package is not installed. Install it using 'pip install accelerate'")
    sys.exit(1)

# Fix Python path for installed packages
sys.path.insert(0, os.path.expanduser("~/.local/lib/python3.10/site-packages"))
sys.path.insert(0, '/opt/conda/lib/python3.10/site-packages')

# Set the device
device = "cpu"  # Change to "cuda" when GPU drivers are updated

# Load and save the model
model_name = "openvla/openvla-7b"
save_dir = os.path.expanduser("~/OpenVLA-forget-tune/models/openvla-7b/")

try:
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        attn_implementation="eager",
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map=device  # Use CPU
    )

    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)
    print(f"✅ Model and processor saved successfully at {save_dir}")

except Exception as e:
    print(f"❌ Error downloading model: {e}")
    sys.exit(1)
