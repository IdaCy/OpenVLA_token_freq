import torch
import json
import os
from transformers import AutoModelForVision2Seq, LlamaTokenizer, AutoImageProcessor
from tqdm import tqdm
import pickle
from PIL import Image
from openvla.prismatic.extern.hf.processing_prismatic import PrismaticProcessor
from collections import defaultdict


# Paths
MODEL_NAME = "models/openvla-7b"  # Load local OpenVLA model
IMAGE_DIR = "data/raw-coco100/train2014"  # Path where images are stored
CAPTIONS_FILE = "data/raw-coco100/captions.json"  # Path to COCO captions
OUTPUT_DIR = "stage1_inf_coco100"  # Output directory for COCO inference

# Ensure output directories exist
for subfolder in ["activations", "attentions", "logits", "predictions"]:
    os.makedirs(os.path.join(OUTPUT_DIR, subfolder), exist_ok=True)

# Detect available device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
print("Loading OpenVLA model...")
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,  # Uses less memory
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to(device)

# Enable hidden states in the model config
model.config.output_hidden_states = True
model.config.output_attentions = True  # Enable attention tracking

# Load tokenizer
TOKENIZER_PATH = f"{MODEL_NAME}/tokenizer.model"
tokenizer = LlamaTokenizer(vocab_file=TOKENIZER_PATH)

# Load image processor manually
print("Loading image processor...")
image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME,
                                                     trust_remote_code=True)

# Initialize OpenVLA's processor manually with image processor
print("Initializing PrismaticProcessor manually...")
processor = PrismaticProcessor(image_processor=image_processor,
                               tokenizer=tokenizer)

# Double-check that everything is correctly attached
if not hasattr(processor, "tokenizer") or processor.tokenizer is None:
    raise ValueError("❌ Tokenizer not properly attached to processor!")
if not hasattr(processor, "image_processor") or processor.image_processor is None:
    raise ValueError("❌ Image processor not properly attached to processor!")

print("Processor, tokenizer, and image processor loaded successfully!")

# Load COCO captions
with open(CAPTIONS_FILE, "r") as f:
    coco_captions = json.load(f)

# Parsing from annotations section
image_caption_map = defaultdict(list)
for item in coco_captions["annotations"]:
    image_caption_map[item["image_id"]].append(item["caption"])

# List all image files in IMAGE_DIR
image_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith(".jpg")])

print(f"Found {len(image_files)} images in {IMAGE_DIR}")

# Run inference
print("Running inference...")
model.eval()

with torch.no_grad():
    for idx, image_filename in enumerate(tqdm(image_files)):        
        # Extract image ID from filename
        img_id = int(image_filename.split("_")[-1].split(".")[0])
        img_path = os.path.join(IMAGE_DIR, image_filename)

        # Check if image has a corresponding caption
        if img_id not in image_caption_map:
            print(f"⚠️ No caption found for {image_filename}. Skipping...")
            with open("missing_captions.log", "a") as log_file:
                log_file.write(f"{image_filename}\n")
            continue

        print(f"📌 Processing image {idx + 1}: {image_filename}")

        # Convert image from file path to PIL Image
        image = Image.open(img_path).convert("RGB")

        if "annotations" not in coco_captions:
            raise ValueError("❌ Error: 'annotations' key missing in COCO captions file!")

        # Retrieve the actual caption
        caption = " ".join(image_caption_map[img_id])

        # Use processor (handles both text + image)
        inputs = processor(text=caption, images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}  # Move to GPU

        # Convert image tensor to `bfloat16` to match model dtype
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

        # Forward pass
        outputs = model(**inputs)

        # Extract required outputs
        hidden_states = outputs.hidden_states if outputs.hidden_states is not None else []
        attentions = outputs.attentions if outputs.attentions is not None else []
        logits = outputs.logits if outputs.logits is not None else torch.tensor([])

        # Capture activations only from selected layers
        selected_layers = [1, 5, 10, 15, 20, 25, 30, 32]
        activations = [hidden_states[i] for i in selected_layers if i < len(hidden_states)]

        # Capture attentions only from selected layers
        selected_attentions = [attentions[i] for i in selected_layers if i < len(attentions)]

        # Decode output logits into text using `generate()`
        generated_tokens = model.generate(
            inputs["input_ids"], 
            max_new_tokens=30  # Allows up to 30 new tokens after input length
        )
        predicted_text = processor.tokenizer.decode(generated_tokens[0],
                                                    skip_special_tokens=True)

        # Define output paths
        output_base = f"img_{img_id}"
        paths = {key: os.path.join(OUTPUT_DIR, key, f"{output_base}_{key}.pkl") for key in 
                 ["activations", "attentions", "logits"]}
        paths["predictions"] = os.path.join(OUTPUT_DIR, "predictions", f"{output_base}_output.txt")

        # Save activations
        if activations:
            with open(paths["activations"], "wb") as f:
                pickle.dump([act.to(torch.float32).cpu().numpy() for act in activations], f)

        # Save attentions
        if selected_attentions:
            with open(paths["attentions"], "wb") as f:
                pickle.dump([att.to(torch.float32).cpu().numpy() for att in selected_attentions], f)

        # Save logits
        with open(paths["logits"], "wb") as f:
            pickle.dump(logits.to(torch.float32).cpu().numpy(), f)

        # Save predicted text output
        with open(paths["predictions"], "w") as f:
            f.write(predicted_text)

print("🎉 Inference complete.")
