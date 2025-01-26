import os
import sys
from transformers import AutoModelForVision2Seq, AutoProcessor
import timm
import numpy as np

import torch
device = "cpu"  # can't do GPU because outdated in HPC! change to "cuda" if the driver is updated
model.to(device)

# Ensure the correct package paths are included
sys.path.insert(0, '/workspace/env/lib/python3.10/site-packages')
sys.path.insert(0, "/workspace/env/lib/python3.10")
sys.path.insert(0, "/workspace/env/lib/python3.10/lib-dynload")
sys.path.insert(0, "/rds/general/user/ifc24/home/anaconda3/envs/hpcenv/lib/python3.12/site-packages")

# Verify package versions
print("Using timm version:", timm.__version__)
print("NumPy version:", np.__version__, "from", np.__file__)

# Check if accelerate is available
try:
    from accelerate import infer_auto_device_map
except ImportError:
    print("❌ The 'accelerate' package is not installed. Install it using 'pip install accelerate'")
    sys.exit(1)

# Set Hugging Face cache directory
HF_CACHE_DIR = os.path.expanduser("~/OpenVLA-forget-tune/cache")
os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE_DIR

# Ensure the cache directory exists
os.makedirs(HF_CACHE_DIR, exist_ok=True)

print(f"✅ Using cache directory: {HF_CACHE_DIR}")

# Display Python environment details
print("Python executable:", sys.executable)
print("Python version:", sys.version)
print("Timm location:", timm.__file__)
print("PYTHONPATH:", sys.path)

# Fixing Python path for installed packages
sys.path.insert(0, os.path.expanduser("~/.local/lib/python3.10/site-packages"))

# Model details
model_name = "openvla/openvla-7b"
save_dir = os.path.expanduser(
    "~/OpenVLA-forget-tune/models_downl-modern-cont/openvla-7b/")

# Model download and saving
try:
    processor = AutoProcessor.from_pretrained(model_name,
                                              trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        attn_implementation="eager",
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map="cpu"  # Force CPU to avoid CUDA issues
    )

    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)
    print(f"✅ Model and processor saved successfully at {save_dir}")

except Exception as e:
    print(f"❌ Error downloading model: {e}")
    sys.exit(1)
