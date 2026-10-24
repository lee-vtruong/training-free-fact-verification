"""Official authenticated snapshot downloader for Qwen3-8B backbone."""
import os
import sys
from pathlib import Path

os.environ["HF_HOME"] = "/home/stackops/whale/cache/huggingface"

from huggingface_hub import snapshot_download

MODEL_ID = "Qwen/Qwen3-8B"
REVISION = "b968826d9c46dd6066d109eabc6255188de91218"

token_file = Path("/home/stackops/whale/cache/huggingface/token")
token = token_file.read_text(encoding="utf-8").strip() if token_file.is_file() else True

print(f"Starting authenticated download of {MODEL_ID} (revision {REVISION}) with 8 parallel workers...")
path = snapshot_download(
    repo_id=MODEL_ID,
    revision=REVISION,
    token=token,
    max_workers=8,
)
print("SUCCESSFULLY DOWNLOADED TO:", path)
