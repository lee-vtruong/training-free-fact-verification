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


def test_m14_pipeline_dry_run():
    """Test full m14_runner pipeline on real server data with SyntheticBackbone to verify all file paths and schemas."""
    import subprocess
    cmd = (
        "source .venv/bin/activate && export PYTHONPATH=src && "
        "R4_PATH=$(find outputs -maxdepth 2 -type f -path 'outputs/m11-store-r14-*/R4.jsonl' | sort | tail -n 1) && "
        "CLAIMS_PATH=$(find outputs -maxdepth 2 -type f -path 'outputs/m11-source-claims-*/claims.jsonl' | sort | tail -n 1) && "
        "TRAIN_JSON='/home/stackops/whale/ERRORFlow/data/raw/averitec/train.json' && "
        "B1_PREDICTIONS=$(find outputs -type f \\( -path '*/B1/predictions.jsonl' -o -path '*b1*preds*/predictions.jsonl' -o -path '*b1*/predictions.jsonl' \\) | sort | tail -n 1) && "
        "python -m tffv.verification.m14_runner "
        "--claims \"$CLAIMS_PATH\" --evidence \"$R4_PATH\" --train-json \"$TRAIN_JSON\" "
        "--b1-predictions \"$B1_PREDICTIONS\" --synthetic --output outputs/m14-preflight-test "
        "--conditions D0 D1"
    )
    print(">>> 3. Testing m14_runner dry-run with real data paths and schemas...")
    res = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
    print("STDOUT:", res.stdout[-500:] if res.stdout else "")
    print("STDERR:", res.stderr[-500:] if res.stderr else "")
    if res.returncode != 0:
        raise RuntimeError(f"Dry-run failed with exit code {res.returncode}")
    print(">>> FULL PIPELINE DRY-RUN VERIFIED ON REAL SERVER DATA!")


if __name__ == "__main__":
    test_backbone_load()
    test_m14_pipeline_dry_run()

