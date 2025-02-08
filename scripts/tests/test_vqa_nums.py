import json

# Paths to JSON files
questions_path = "data/raw-vqa/v2_OpenEnded_mscoco_train2014_questions.json"
annotations_path = "data/raw-vqa/v2_mscoco_train2014_annotations.json"

# Load and count questions
with open(questions_path, "r") as f:
    questions_data = json.load(f)
num_questions = len(questions_data["questions"])

# Load and count answers
with open(annotations_path, "r") as f:
    annotations_data = json.load(f)
num_answers = len(annotations_data["annotations"])

# Print results
print(f"Total Questions: {num_questions}")
print(f"Total Answers: {num_answers}")

# Check if they match
assert num_questions == num_answers, "Mismatch between number of questions and answers!"
