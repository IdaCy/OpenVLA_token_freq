import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image

import os
os.environ['TRANSFORMERS_CACHE'] = '/mnt/cache/huggingface'
os.environ['HF_HOME'] = '/mnt/cache/huggingface'
os.environ['TORCH_HOME'] = '/mnt/cache/torch'
os.environ['HF_DATASETS_CACHE'] = '/mnt/cache/huggingface_datasets'

model_path = '/mnt/OpenVLA-forget-tune/models_downl-modern-cont/openvla-7b/'
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(model_path, trust_remote_code=True)
print('Model loaded successfully!')

# Set model directory path
model_path = "/rds/general/user/ifc24/home/OpenVLA-forget-tune/models_downl-modern-cont/openvla-7b/"

# Load processor and model
try:
    print("Loading processor and model...")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,  # Use BF16 to reduce memory
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    # Load a sample image (replace with actual image path)
    image_path = "/rds/general/user/ifc24/home/sample_image.jpg"
    image = Image.open(image_path)

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
