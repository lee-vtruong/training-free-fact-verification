"""Explicit non-neural test doubles. NEVER an implementation of Qwen retrieval."""
import numpy as np
from tffv.provenance import digest
from tffv.retrieval.data import tokens


class SyntheticEncoder:
    def __init__(self):
        self.forward_passes = self.token_count = self.truncated_inputs = 0
        self.document_rows_encoded = 0

    def encode(self, texts, query=False):
        rows = []
        for text in texts:
            vector = np.zeros(1024, dtype=np.float32)
            words = tokens(text)
            for word in words:
                vector[int(digest(word)[:8], 16) % 1024] += 1
            norm = np.linalg.norm(vector)
            if not norm: raise ValueError("Empty synthetic text")
            rows.append(vector / norm)
            self.token_count += len(words)
        self.forward_passes += 1
        self.document_rows_encoded += 0 if query else len(texts)
        return np.asarray(rows, dtype=np.float32)

    def sync(self): pass
    def reset_peak(self): pass
    def peak_bytes(self): return None
    def close(self): pass


class SyntheticReranker(SyntheticEncoder):
    def score(self, claim, texts):
        self.forward_passes += 1
        self.token_count += sum(len(tokens(text)) + len(tokens(claim)) for text in texts)
        query = set(tokens(claim))
        return [len(query & set(tokens(text))) / max(1, len(query)) for text in texts]
