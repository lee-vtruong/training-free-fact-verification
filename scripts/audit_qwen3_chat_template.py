"""Audit Qwen3-8B chat template, thinking behavior, and tokenization."""
import sys
from pathlib import Path
from transformers import AutoTokenizer

MODEL_PATH = "/home/stackops/whale/cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"

tok = AutoTokenizer.from_pretrained(MODEL_PATH)
print("=== TOKENIZER INFO ===")
print("Vocab size:", len(tok))
print("EOS token:", tok.eos_token, "id:", tok.eos_token_id)
print("PAD token:", tok.pad_token, "id:", tok.pad_token_id)
print("Special tokens:", tok.all_special_tokens[:10])

print("\n=== CHAT TEMPLATE ===")
print(tok.chat_template)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Verify this claim."},
]

# Test standard apply_chat_template
formatted_default = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print("\n=== DEFAULT FORMATTED PROMPT ===")
print(repr(formatted_default))

# Test enable_thinking if supported
for flag in [True, False]:
    try:
        f = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=flag)
        print(f"\n=== enable_thinking={flag} SUPPORTED! ===")
        print(repr(f))
    except Exception as e:
        print(f"\n=== enable_thinking={flag} FAILED: {e}")
