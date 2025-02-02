import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import os

# Define full paths instead of using '~'
BASE_PATH = "/rds/general/user/ifc24/home/OpenVLA-forget-tune"
CACHE_PATH = os.path.join(BASE_PATH, "models/openvla-7b/cache")
MODEL_PATH = os.path.join(BASE_PATH, "models/openvla-7b")
IMAGE_PATH = os.path.join(BASE_PATH, "data/images/cat.jpg")

# Set environment variables for cache locations
os.environ['TRANSFORMERS_CACHE'] = os.path.join(CACHE_PATH, "huggingface")
os.environ['HF_HOME'] = os.path.join(CACHE_PATH, "huggingface")
os.environ['TORCH_HOME'] = os.path.join(CACHE_PATH, "torch")
os.environ['HF_DATASETS_CACHE'] = os.path.join(CACHE_PATH, "huggingface_datasets")

# Load processor and model
try:
    print("Loading processor and model...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,  # Use BF16 to reduce memory
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    # Load a sample image
    image = Image.open(IMAGE_PATH)

    # Define a test prompt
    prompt = "Describe this image."

    # Prepare inputs
    inputs = processor(prompt, image, return_tensors="pt").to(model.device, dtype=torch.bfloat16)

    # Run inference
    print("Running inference...")
    with torch.no_grad():
        outputs = model.generate(**inputs)

    # Decode output
    generated_text = processor.decode(outputs[0], skip_special_tokens=True)
    print("Generated output:", generated_text)

except Exception as e:
    print(f"❌ Model inference failed: {e}")
