import torch
from transformers import AutoModelForVision2Seq, AutoProcessor

model_path = "/rds/general/user/ifc24/home/OpenVLA-forget-tune/models_downl-modern-cont/openvla-7b/"

try:
    # Add trust_remote_code=True to allow loading custom code
    processor = AutoProcessor.from_pretrained(model_path,
                                              trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(model_path,
                                                   trust_remote_code=True)

    input_text = "Describe this image"
    inputs = processor(text=input_text, return_tensors="pt").to("cpu")

    with torch.no_grad():
        outputs = model.generate(**inputs)

    print("Generated output:", processor.decode(outputs[0],
                                                skip_special_tokens=True))
    print("✅ Model inference successful!")

except Exception as e:
    print(f"❌ Model inference failed: {e}")
