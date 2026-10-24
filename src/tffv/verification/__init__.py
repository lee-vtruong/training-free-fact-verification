"""Training-Free Fact Verification (M1.2).

Frozen backbone zero-shot verification, error profiling, and group safety.
"""
from tffv.verification.ontology import VERDICTS, normalize_verdict

__all__ = ["VERDICTS", "normalize_verdict"]
