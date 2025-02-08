import torch
import os
import pickle
from transformers import AutoModelForVision2Seq, LlamaTokenizer, AutoImageProcessor
from tqdm import tqdm
from PIL import Image
from openvla.prismatic.extern.hf.processing_prismatic import PrismaticProcessor

# **1️⃣ Define Paths**
MODEL_NAME = "models/openvla-7b"
ROOT_DIR = "data/raw-bridge/scripted_raw"  # New path for bridge dataset
OUTPUT_DIR = "stage1_inf_bridge_pred"

# Ensure output directories exist
for subfolder in ["activations", "attentions", "logits", "representations", "probabilities", "predictions"]:
    os.makedirs(os.path.join(OUTPUT_DIR, subfolder), exist_ok=True)

# **2️⃣ Detect Available Device**
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# **3️⃣ Load Model**
print("🔄 Loading OpenVLA model...")
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to(device)

# Enable hidden states in the model config
model.config.output_hidden_states = True
model.config.output_attentions = True

# **4️⃣ Load Tokenizer**
TOKENIZER_PATH = f"{MODEL_NAME}/tokenizer.model"
tokenizer = LlamaTokenizer(vocab_file=TOKENIZER_PATH)

# **5️⃣ Load Image Processor**
print("📸 Loading image processor...")
image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
processor = PrismaticProcessor(image_processor=image_processor, tokenizer=tokenizer)

print("✅ Model, tokenizer, and image processor loaded successfully!")

# **6️⃣ Recursively Find All Images in Bridge Dataset**
image_files = []
for root, _, files in os.walk(ROOT_DIR):
    for file in files:
        if file.endswith((".jpg", ".png")):
            full_path = os.path.join(root, file)
            image_files.append(full_path)

if not image_files:
    print("❌ No images found in", ROOT_DIR)
    exit()

print(f"📂 Found {len(image_files)} images in {ROOT_DIR}")

# **7️⃣ Run Inference**
print("🚀 Running inference on Bridge dataset...")
model.eval()
with torch.no_grad():
    for idx, img_path in enumerate(tqdm(image_files[:1000])):  # Process 5000 images
        img_filename = os.path.basename(img_path)
        rel_path = os.path.relpath(img_path, ROOT_DIR)  # Relative path for naming
        unique_filename = rel_path.replace("/", "_")  # Convert to flat filename

        print(f"📌 Processing image {idx + 1}/{len(image_files)}: {unique_filename}")

        # **Load Image**
        image = Image.open(img_path).convert("RGB")

        # **Prepare Inputs**
        prompt = "Describe the robot's action in this image."
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Convert image tensor to `bfloat16` (match model dtype)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

        # **Run Model**
        outputs = model(**inputs)

        # **Extract Outputs**
        hidden_states = outputs.hidden_states if outputs.hidden_states else []
        attentions = outputs.attentions if outputs.attentions else []
        logits = outputs.logits if outputs.logits is not None else torch.tensor([])

        # Compute Softmax Probabilities
        probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy() if logits.numel() > 0 else None

        # **Generate Caption**
        generated_tokens = model.generate(
            inputs["input_ids"],
            max_new_tokens=30  # Generate up to 30 tokens
        )

        predicted_text = processor.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

        print(f"📝 Generated caption: {predicted_text}")

        # **8️⃣ Save Outputs**
        paths = {key: os.path.join(OUTPUT_DIR, key, f"{unique_filename}_{key}.pkl") for key in 
                 ["activations", "attentions", "logits", "representations", "probabilities"]}
        paths["predictions"] = os.path.join(OUTPUT_DIR, "predictions", f"{unique_filename}_output.txt")

        # Save Intermediate Outputs
        if hidden_states:
            with open(paths["activations"], "wb") as f:
                pickle.dump([hs.to(torch.float32).cpu().numpy() for hs in hidden_states], f)

        if attentions:
            with open(paths["attentions"], "wb") as f:
                pickle.dump([att.to(torch.float32).cpu().numpy() for att in attentions], f)

        with open(paths["logits"], "wb") as f:
            pickle.dump(logits.to(torch.float32).cpu().numpy(), f)

        if hidden_states:
            with open(paths["representations"], "wb") as f:
                pickle.dump(hidden_states[-1].to(torch.float32).cpu().numpy(), f)

        if probs is not None:
            with open(paths["probabilities"], "wb") as f:
                pickle.dump(probs, f)

        with open(paths["predictions"], "w") as f:
            f.write(predicted_text)

        # Log every 1000 images
        if (idx + 1) % 1000 == 0:
            print(f"💾 Saved checkpoint at {unique_filename}.")

print("🎉 Inference complete on Bridge dataset!")
