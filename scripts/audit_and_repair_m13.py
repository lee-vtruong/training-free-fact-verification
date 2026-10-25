"""CLI runner for M1.3 comprehensive audit and repair."""
import sys
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tffv.verification.m13_audit import run_comprehensive_audit_and_repair

if __name__ == "__main__":
    run_comprehensive_audit_and_repair()
