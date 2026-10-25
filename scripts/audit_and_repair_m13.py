"""CLI runner for M1.3 comprehensive audit and repair."""
import argparse
import sys
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tffv.verification.m13_audit import run_comprehensive_audit_and_repair

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit and repair M1.3 evaluation reporting.")
    parser.add_argument("--train-json", type=Path, default=None, help="Path to AVeriTeC train.json")
    parser.add_argument("--m13-dir", type=Path, default=None, help="Path to M1.3 run output directory")
    parser.add_argument("--tokenizer-dir", type=Path, default=None, help="Path to Qwen3 tokenizer or snapshot")
    parser.add_argument("--output-dir", type=Path, default=None, help="Path to write repaired canonical artifacts")
    args = parser.parse_args()

    run_comprehensive_audit_and_repair(
        train_json_path=args.train_json,
        m13_dir=args.m13_dir,
        tokenizer_dir=args.tokenizer_dir,
        output_dir=args.output_dir,
    )

