"""Strict canonical data contracts. Dataset-specific adapters require inspection."""
from .records import load_audit_inputs, read_jsonl

__all__ = ["load_audit_inputs", "read_jsonl"]
