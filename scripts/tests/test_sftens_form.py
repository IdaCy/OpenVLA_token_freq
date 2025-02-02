from safetensors import safe_open

model_files = [
    "/rds/general/user/ifc24/home/OpenVLA-forget-tune/models/openvla-7b/model-00001-of-00004.safetensors",
    "/rds/general/user/ifc24/home/OpenVLA-forget-tune/models/openvla-7b/model-00002-of-00004.safetensors",
    "/rds/general/user/ifc24/home/OpenVLA-forget-tune/models/openvla-7b/model-00003-of-00004.safetensors",
    "/rds/general/user/ifc24/home/OpenVLA-forget-tune/models/openvla-7b/model-00004-of-00004.safetensors"
]

for file in model_files:
    try:
        with safe_open(file, framework="pt", device="cpu") as f:
            print(f"✅ {file} is valid!")
    except Exception as e:
        print(f"❌ {file} is corrupted: {e}")
