import json
import os

# Paths to subset files
questions_path = "data/raw-vqa/subset_questions.json"
answers_path = "data/raw-vqa/subset_answers.json"
images_dir = "data/raw-vqa/train2014"

# Check JSON file contents
with open(questions_path, 'r') as q_file, open(answers_path, 'r') as a_file:
    questions = json.load(q_file)
    answers = json.load(a_file)

print(f"Number of questions: {len(questions)}")
print(f"Number of answers: {len(answers)}")

# Verify images count
image_count = len(os.listdir(images_dir))
print(f"Number of images: {image_count}")

if len(questions) == len(answers) == 10000 and image_count == 10000:
    print("VQA subset verified successfully.")
else:
    print("VQA subset verification failed.")
