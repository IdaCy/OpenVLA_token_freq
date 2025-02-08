import sentencepiece as spm

MODEL_PATH = "models/openvla-7b/tokenizer.model"

sp = spm.SentencePieceProcessor()
if sp.Load(MODEL_PATH):
    print("✅ SentencePiece tokenizer loaded successfully!")
    print("Vocabulary size:", sp.GetPieceSize())
else:
    print("❌ Failed to load SentencePiece tokenizer!")
