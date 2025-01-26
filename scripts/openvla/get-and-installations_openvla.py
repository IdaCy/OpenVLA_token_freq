import os
import sys
import subprocess
import importlib
import argparse
import shutil
import torch
from transformers.utils.hub import cached_download
from transformers import AutoModelForVision2Seq, AutoProcessor

# Ensure the correct Conda environment is activated
conda_env_path = "/rds/general/user/ifc24/home/anaconda3/etc/profile.d/conda.sh"
os.system(f"source {conda_env_path} && conda activate hpcenv")

# Prioritize the correct Conda environment path and remove conflicting paths
sys.path = [p for p in sys.path if "conda/lib/python3.10" not in p and "workspace/env" not in p]
sys.path.insert(0, "/rds/general/user/ifc24/home/anaconda3/envs/hpcenv/lib/python3.12/site-packages")

# Required dependencies with expected versions
required_packages = {
    "timm": "0.9.10",
    "numpy": "2.2.1",
    "transformers": ">=4.0.0",
    "huggingface_hub": ">=0.15.1",
    "torch": "2.2.1",
    "accelerate": ">=0.27.2",
    "mkl": ">=2023.2.0",  # MKL required for numpy, torch
}

# Command-line argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--check-only', action='store_true', help="Only check dependencies, do not install")
args = parser.parse_args()

def install_package(package):
    """ Install a missing or incorrect package using Conda. """
    try:
        if package == "mkl":
            print(f"🔄 Installing {package} using Conda...")
            subprocess.check_call(["conda", "install", "-y", "-c", "conda-forge", "mkl", "mkl-include", "mkl-service"])
        else:
            print(f"🔄 Installing {package} using Conda...")
            subprocess.check_call(["conda", "install", "-y", "-c", "conda-forge", f"{package}=={required_packages[package]}"])
        print(f"✅ {package} installed successfully.")
    except Exception as e:
        print(f"❌ Failed to install {package} with Conda: {e}")
        sys.exit(1)

def check_and_install_packages():
    """ Check if required packages are installed with the correct versions. """
    for package, expected_version in required_packages.items():
        try:
            module = importlib.import_module(package)
            installed_version = module.__version__
            print(f"✅ {package}: Found version {installed_version}")

            # If the version doesn't match, reinstall it
            if installed_version != expected_version and not args.check_only:
                print(f"⚠️ {package} version mismatch. Expected {expected_version}, but found {installed_version}")
                install_package(f"{package}=={expected_version}")

        except ImportError:
            print(f"❌ {package} not found.")
            if not args.check_only:
                install_package(f"{package}=={expected_version}")

check_and_install_packages()

# Confirm paths to ensure correct dependencies are being loaded
print("\nPython executable:", sys.executable)
print("PYTHONPATH:", sys.path)

# Set Hugging Face cache directory
os.environ["HF_HOME"] = os.path.expanduser("~/OpenVLA-forget-tune/cache")
os.environ["TRANSFORMERS_CACHE"] = os.environ["HF_HOME"]

# Ensure the cache directory exists
os.makedirs(os.environ["HF_HOME"], exist_ok=True)
print(f"Using cache directory: {os.environ['HF_HOME']}")

# Check disk space before proceeding (minimum 20GB required)
total, used, free = shutil.disk_usage("/")
if free < 20 * (1024 ** 3):  # Check if at least 20GB free
    print("❌ Insufficient disk space for model download")
    sys.exit(1)

# Test imports after ensuring dependencies
import timm
print("Using timm version:", timm.__version__)

import numpy
print(numpy.__version__, numpy.__file__)

# Check for GPU availability and set device accordingly
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {device}")

# Model download and processing section
model_name = "openvla/openvla-7b"
save_dir = os.path.expanduser("~/OpenVLA-forget-tune/models/openvla-7b/")

def download_with_retry(url, retries=3):
    """Download model with retry logic in case of failure."""
    for attempt in range(retries):
        try:
            cached_download(url)
            print("✅ Model downloaded successfully.")
            return
        except Exception as e:
            print(f"⚠️ Download failed (attempt {attempt+1}): {e}")
    print("❌ Failed to download model after multiple attempts.")
    sys.exit(1)

try:
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        attn_implementation="eager",
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map=device  # Automatically select GPU or CPU
    )

    # Save model locally
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)
    print(f"✅ Model and processor saved successfully at {save_dir}")

except Exception as e:
    print(f"❌ Error downloading model: {e}")
