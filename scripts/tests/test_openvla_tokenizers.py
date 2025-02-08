from transformers import LlamaTokenizer

MODEL_NAME = "models/openvla-7b"
TOKENIZER_PATH = f"{MODEL_NAME}/tokenizer.model"

try:
    tokenizer = LlamaTokenizer(vocab_file=TOKENIZER_PATH)
    print("✅ Tokenizer loaded successfully!")
    print("Special tokens:", tokenizer.special_tokens_map)
except Exception as e:
    print("❌ Tokenizer loading failed:", e)
