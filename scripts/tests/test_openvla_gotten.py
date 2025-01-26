import torch
from transformers import AutoModelForVision2Seq, AutoProcessor

# Set model path
model_path = "/rds/general/user/ifc24/home/OpenVLA-forget-tune/models/openvla-7b/"

try:
    # Load the processor
    processor = AutoProcessor.from_pretrained(model_path)
    print("✅ Processor loaded successfully.")

    # Load the model
    model = AutoModelForVision2Seq.from_pretrained(model_path)
    print("✅ Model loaded successfully.")

    # Check model device compatibility
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"✅ Model moved to {device} successfully.")

    # Perform a simple forward pass
    dummy_input = torch.rand(1, 3, 224, 224).to(device)
    with torch.no_grad():
        output = model(dummy_input)
    print("✅ Model forward pass successful.")

except Exception as e:
    print(f"❌ Model loading failed: {e}")
