import json
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# Paths to the VQA data
IMAGE_DIR = "data/raw-vqa/train2014/"
QUESTIONS_PATH = "data/raw-vqa/subset_questions.json"
ANSWERS_PATH = "data/raw-vqa/subset_answers.json"
OUTPUT_PATH = "results/vqa_predictions.json"

# Load the model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-vqa-base").to("cuda")

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

    # Generate answer
    output = model.generate(**inputs)
    answer = processor.decode(output[0], skip_special_tokens=True)

    results.append({"image_id": image_id, "question": question,
                    "answer": answer})

    print(f"Processed {image_id}: {question} -> {answer}")

# Save results
with open(OUTPUT_PATH, "w") as f:
    json.dump(results, f, indent=4)

print("Inference completed. Results saved to:", OUTPUT_PATH)
