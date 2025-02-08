import torch
import json
import os
from transformers import AutoModelForVision2Seq, LlamaTokenizer, AutoImageProcessor
from tqdm import tqdm
import pickle
from PIL import Image
from openvla.prismatic.extern.hf.processing_prismatic import PrismaticProcessor

# Paths
MODEL_NAME = "models/openvla-7b"  # Load local OpenVLA model
DATA_DIR = "data/raw-vqa100"
IMAGE_DIR = os.path.join(DATA_DIR, "train2014")
QUESTION_FILE = os.path.join(DATA_DIR, "v2_OpenEnded_mscoco_train2014_questions.json")
ANNOTATION_FILE = os.path.join(DATA_DIR, "v2_mscoco_train2014_annotations.json")
OUTPUT_DIR = "stage1_inf_vqa100"

# Ensure output directories exist
for subfolder in ["activations", "attentions", "logits", "predictions"]:
    os.makedirs(os.path.join(OUTPUT_DIR, subfolder), exist_ok=True)

# Detect available device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
print("Loading OpenVLA model...")
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to(device)

# Enable required outputs in the model config
model.config.output_hidden_states = True
model.config.output_attentions = True

# Load tokenizer
TOKENIZER_PATH = f"{MODEL_NAME}/tokenizer.model"
try:
    tokenizer = LlamaTokenizer(vocab_file=TOKENIZER_PATH)
    print("✅ Tokenizer loaded successfully!")
    print("Special tokens:", tokenizer.special_tokens_map)
except Exception as e:
    print(f"❌ Tokenizer loading failed: {e}")
    exit(1)

# Load image processor
print("Loading image processor...")
image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)

# Initialize OpenVLA's processor manually
print("Initializing PrismaticProcessor manually...")
processor = PrismaticProcessor(image_processor=image_processor, tokenizer=tokenizer)

# Double-check processor attachments
if not hasattr(processor, "tokenizer") or processor.tokenizer is None:
    raise ValueError("❌ Tokenizer not properly attached to processor!")
if not hasattr(processor, "image_processor") or processor.image_processor is None:
    raise ValueError("❌ Image processor not properly attached to processor!")

print("Processor, tokenizer, and image processor loaded successfully!")

# Load VQA questions
print("Loading VQA V2 dataset...")
with open(QUESTION_FILE, "r") as f:
    vqa_questions = json.load(f)["questions"]

print(f"Loaded {len(vqa_questions)} questions.")

# Load VQA answers
print("Loading VQA annotations...")
with open(ANNOTATION_FILE, "r") as f:
    vqa_annotations = json.load(f)["annotations"]

print(f"Loaded {len(vqa_annotations)} annotations.")

# Index questions & answers by image_id for fast lookup
question_map = {}
for q in vqa_questions:
    question_map.setdefault(q["image_id"], []).append(q["question"])

answer_map = {}
for annotation in vqa_annotations:
    img_id = annotation["image_id"]
    answer_map.setdefault(img_id, []).append(annotation["answers"])

# List all images in the dataset folder
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

        # Check if there are associated questions and answers
        if img_id not in question_map or img_id not in answer_map:
            print(f"⚠️ WARNING: No questions/answers found for image {image_filename}. Skipping.")
            continue  # Skip images without questions/answers

        print(f"📌 Processing image {idx + 1}: {image_filename}")

        # Convert image from file path to PIL Image
        image = Image.open(img_path).convert("RGB")

        for question in question_map[img_id]:  # Process all Qs for image
            # Use processor (handles both text + image)
            inputs = processor(text=question, images=image, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}  # -> GPU

            # Convert image tensor to `bfloat16` to match model dtype
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

            # Forward pass
            outputs = model(**inputs)

            # Extract selected layers' activations and attentions
            hidden_states = outputs.hidden_states if outputs.hidden_states is not None else []
            attentions = outputs.attentions if outputs.attentions is not None else []
            logits = outputs.logits if outputs.logits is not None else torch.tensor([])

            # Keep only specific layers: 1, 5, 10, 15, 20, 25, 30, 32
            selected_layers = [0, 4, 9, 14, 19, 24, 29, 31]  # Python indexing (zero-based)
            hidden_states_selected = [hidden_states[i] for i in selected_layers] if hidden_states else []
            attentions_selected = [attentions[i] for i in selected_layers] if attentions else []

            # Decode output logits into text using `generate()`
            generated_tokens = model.generate(
                inputs["input_ids"], 
                max_new_tokens=30  # up to 30 new tokens after input length
            )

            # Debug: Print token IDs before decoding
            print(f"Generated token IDs for image {img_id}: {generated_tokens[0].tolist()}")

            try:
                predicted_text = processor.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
            except Exception as e:
                print(f"❌ Token decoding failed: {e}")
                predicted_text = "[ERROR: Decoding Failed]"

            # Define output paths
            output_base = f"img_{img_id}_q_{hash(question) % (10**8)}"
            paths = {key: os.path.join(OUTPUT_DIR, key, f"{output_base}_{key}.pkl") for key in 
                     ["activations", "attentions", "logits"]}
            paths["predictions"] = os.path.join(OUTPUT_DIR, "predictions", f"{output_base}_output.txt")

            # Save activations
            if hidden_states_selected:
                with open(paths["activations"], "wb") as f:
                    pickle.dump([hs.to(torch.float32).cpu().numpy() for hs in hidden_states_selected], f)

            # Save attentions
            if attentions_selected:
                with open(paths["attentions"], "wb") as f:
                    pickle.dump([att.to(torch.float32).cpu().numpy() for att in attentions_selected], f)

            # Save logits
            with open(paths["logits"], "wb") as f:
                pickle.dump(logits.to(torch.float32).cpu().numpy(), f)

            # Save predicted text output
            with open(paths["predictions"], "w") as f:
                f.write(predicted_text)

print("🎉 Inference complete.")
