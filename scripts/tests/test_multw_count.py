import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "models/models--openvla--openvla-7b"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path).cuda()

# Sample input
sample_input = "User: I'd like to book a hotel for tonight.\nSystem:"
inputs = tokenizer(sample_input, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model(**inputs)
    print(outputs.logits.shape)  # Ensure logits are returned
