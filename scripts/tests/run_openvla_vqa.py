import json
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

# Load OpenVLA model and processor
MODEL_NAME = "OpenVLA/vqa-large"
IMAGE_DIR = "data/raw-vqa/train2014/"
QUESTIONS_PATH = "data/raw-vqa/subset_questions.json"
OUTPUT_PATH = "results/openvla_vqa_results.json"

# Load model and processor
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForVision2Seq.from_pretrained(MODEL_NAME).to("cuda")

# Load questions
with open(QUESTIONS_PATH, "r") as f:
    questions_data = json.load(f)

# Run inference
results = []
for item in questions_data:
    image_id = item["image_id"]
    question = item["question"]
    image_path = f"{IMAGE_DIR}/COCO_train2014_{str(image_id).zfill(12)}.jpg"

    # Load and process image
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, question, return_tensors="pt").to("cuda")

    # Generate answer using OpenVLA
    with torch.no_grad():
        outputs = model.generate(**inputs)

    answer = processor.decode(outputs[0], skip_special_tokens=True)

    results.append({"image_id": image_id, "question": question,
                    "answer": answer})
    print(f"Processed {image_id}: {question} -> {answer}")

# Save results
with open(OUTPUT_PATH, "w") as f:
    json.dump(results, f, indent=4)

print("Inference completed. Results saved to:", OUTPUT_PATH)
