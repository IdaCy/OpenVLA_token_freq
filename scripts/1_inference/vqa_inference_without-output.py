import torch
import json
import os
from transformers import AutoModelForVision2Seq, LlamaTokenizer
from transformers import AutoImageProcessor
from tqdm import tqdm
import pickle
from PIL import Image
from openvla.prismatic.extern.hf.processing_prismatic import PrismaticProcessor

# Paths
MODEL_NAME = "models/openvla-7b"  # Load local OpenVLA model
DATA_DIR = "data/raw-vqa"
IMAGE_DIR = os.path.join(DATA_DIR, "train2014")
QUESTION_FILE = os.path.join(DATA_DIR,
                             "v2_OpenEnded_mscoco_train2014_questions.json")
OUTPUT_DIR = "stage1_inf_vqa"

# Ensure output directories exist
for subfolder in ["activations", "attentions", "logits", "representations",
                  "probabilities"]:
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
model.config.output_attentions = True  # Optional, if needed

# Load tokenizer using the **working approach**
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

# Tokenization Test (Ensure everything works before inference)
test_input = "What is the robot doing?"
tokens = processor.tokenizer(test_input, return_tensors="pt")
print("📝 Test tokenized input IDs:", tokens["input_ids"])  # only token IDs

print("Processor's tokenizer special tokens:",
      processor.tokenizer.special_tokens_map)

# Load VQA questions
print("Loading VQA V2 dataset...")
with open(QUESTION_FILE, "r") as f:
    vqa_questions = json.load(f)["questions"]

print(f"Loaded {len(vqa_questions)} questions.")

# Run inference
print("Running inference...")
model.eval()
with torch.no_grad():
    for idx, question in enumerate(tqdm(vqa_questions[:1000])):
        img_id = question["image_id"]
        img_filename = f"train2014/COCO_train2014_{img_id:012d}.jpg"
        img_path = os.path.join(IMAGE_DIR, img_filename)

        if not os.path.exists(img_path):
            print(f"⚠️ WARNING: Image {img_filename} not found! " +
                  f"Skipping question {idx}.")
            continue  # Skip missing images

        print(f"📌 Processing question {idx + 1}: {question['question']}")

        # Convert image from file path to PIL Image
        image = Image.open(img_path).convert("RGB")

        # Use processor (handles both text + image)
        inputs = processor(text=question["question"], images=image,
                           return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}  # Move to GPU

        # Convert image tensor to `bfloat16` to match model dtype
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

        # Forward pass
        outputs = model(**inputs)

        # Ensure hidden_states exist (avoid NoneType error)
        hidden_states = outputs.hidden_states if (
            outputs.hidden_states) is not None else []
        attentions = outputs.attentions if (
            outputs.attentions) is not None else []
        logits = outputs.logits if (
            outputs.logits) is not None else torch.tensor([])

        # Compute softmax probabilities
        probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy(
            ) if logits.numel() > 0 else None

        # Save structured outputs
        output_prefix = os.path.join(OUTPUT_DIR, f"img_{img_id}")

        # Save activations
        if hidden_states:
            with open(f"{output_prefix}_activations.pkl", "wb") as f:
                pickle.dump([hs.to(dtype=torch.float16).cpu().numpy(
                    ) for hs in hidden_states], f)

        # Save attentions
        if attentions:
            with open(f"{output_prefix}_attentions.pkl", "wb") as f:
                pickle.dump([att.to(dtype=torch.float16).cpu().numpy(
                    ) for att in attentions], f)

        # Save logits
        with open(f"{output_prefix}_logits.pkl", "wb") as f:
            pickle.dump(logits.to(dtype=torch.float16).cpu().numpy(), f)

        # Save token embeddings (representation shift)
        if hidden_states:
            with open(f"{output_prefix}_representations.pkl", "wb") as f:
                pickle.dump(hidden_states[-1].to(dtype=torch.float16).cpu(
                    ).numpy(), f)

        # Save probabilities
        if probs is not None:
            with open(f"{output_prefix}_probabilities.pkl", "wb") as f:
                pickle.dump(probs, f)

        # Log every 500 samples
        if (idx + 1) % 500 == 0:
            print(f"💾 Saved checkpoint at {output_prefix}.")

print("🎉 Inference complete.")
