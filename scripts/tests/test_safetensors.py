from transformers import AutoModelForVision2Seq

model_path = "/rds/general/user/ifc24/home/OpenVLA-forget-tune/models/openvla-7b/"
model = AutoModelForVision2Seq.from_pretrained(model_path, trust_remote_code=True)
print("✅ Model successfully loaded!")

