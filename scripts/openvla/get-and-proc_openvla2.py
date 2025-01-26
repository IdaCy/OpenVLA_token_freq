import os
import sys
sys.path.append('/opt/conda/lib/python3.10/site-packages')
sys.path.insert(0, '/workspace/env/lib/python3.10/site-packages')
sys.path.insert(0, "/workspace/env/lib/python3.10")
sys.path.insert(0, "/workspace/env/lib/python3.10/lib-dynload")
sys.path.insert(0, "/rds/general/user/ifc24/home/anaconda3/envs/hpcenv/lib/python3.12/site-packages")

import timm
print("Using timm version:", timm.__version__)

import numpy
print(numpy.__version__, numpy.__file__)

from transformers import AutoModelForVision2Seq, AutoProcessor

try:
    from accelerate import infer_auto_device_map
except ImportError:
    print("The 'accelerate' package is not installed. Install it using 'pip install accelerate'")
    sys.exit(1)

# Force Hugging Face to use a writable cache directory
os.environ["HF_HOME"] = os.path.expanduser("~/OpenVLA-forget-tune/cache")
os.environ["TRANSFORMERS_CACHE"] = os.environ["HF_HOME"]

# Ensure the cache directory exists
os.makedirs(os.environ["HF_HOME"], exist_ok=True)

print(f"Using cache directory: {os.environ['HF_HOME']}")

print("Using Python from:", sys.executable)
print("NumPy version:", numpy.__version__, "from", numpy.__file__)
print("PYTHONPATH:", sys.path)


# Fixing Python path for installed packages
sys.path.insert(0, os.path.expanduser("~/.local/lib/python3.10/site-packages"))

model_name = "openvla/openvla-7b"
save_dir = os.path.expanduser("~/OpenVLA-forget-tune/models2/openvla-7b/")

try:
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        attn_implementation="eager",
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map="cpu"  # Force CPU to avoid CUDA issues
    )

    # Save model locally
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)
    print(f"Model and processor saved successfully at {save_dir}")

except Exception as e:
    print(f"Error downloading model: {e}")
