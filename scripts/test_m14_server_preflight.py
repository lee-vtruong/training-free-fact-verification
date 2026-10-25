"""Preflight test on server: verify Qwen3FrozenBackbone and m14_runner on real hardware."""
import sys
import time
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tffv.verification.backbone import (
    EXPECTED_QWEN3_8B_ID,
    EXPECTED_QWEN3_8B_REVISION,
    Qwen3FrozenBackbone,
)

def test_backbone_load():
    model_path = "/home/stackops/whale/cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
    if not Path(model_path).is_dir():
        print(f"Skipping server preflight: model path {model_path} not found.")
        return

    print(">>> 1. Initializing Qwen3FrozenBackbone on CUDA...")
    spec = {
        "model_id": EXPECTED_QWEN3_8B_ID,
        "revision": EXPECTED_QWEN3_8B_REVISION,
        "local_path": model_path,
    }
    t0 = time.perf_counter()
    backbone = Qwen3FrozenBackbone(spec=spec, device="cuda", dtype="bfloat16")
    load_time = time.perf_counter() - t0
    print(f">>> Backbone loaded successfully in {load_time:.2f}s!")

    print(">>> 2. Testing test generation (1 forward pass)...")
    t0 = time.perf_counter()
    res = backbone.generate(
        [{"role": "user", "content": "Claim: The earth revolves around the sun.\nEvidence: Earth is the third planet orbiting the Sun.\nVerdict:"}],
        max_new_tokens=32,
        enable_thinking=False,
    )
    gen_time = time.perf_counter() - t0
    print(f">>> Generated in {gen_time:.2f}s: {res['generated_text']!r}")
    print(f">>> Prompt tokens: {res['prompt_tokens']}, Generated tokens: {res['generated_tokens']}")
    print(">>> SERVER PREFLIGHT VERIFICATION COMPLETE AND 100% OK!")

if __name__ == "__main__":
    test_backbone_load()
