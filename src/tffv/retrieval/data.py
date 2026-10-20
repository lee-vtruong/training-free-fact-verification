"""Label-free retrieval contracts; official QA/justifications are not a corpus."""
import re
from dataclasses import asdict, dataclass
from urllib.parse import urlsplit
from tffv.data import read_jsonl
from tffv.provenance import digest, file_hash


ONTOLOGY = ["SUPPORTED", "REFUTED", "NOT ENOUGH INFO", "CONFLICTING"]
CLAIM_FIELDS = {"sample_id", "dataset", "split", "claim"}
DOCUMENT_FIELDS = {"document_id", "text", "source_url", "source_kind", "retrieved_at", "acquisition_id"}
VERDICT = re.compile(r"\b(?:verdict|fact[ -]?check(?:ing)? rating|gold[_ ]label)\s*[:=]|\b(?:we rate|rated (?:this|the) claim|the claim is (?:false|true))\b", re.I)


def normalize(text):
    return " ".join(text.casefold().split())


def tokens(text):
    return re.findall(r"\w+", text.casefold(), re.UNICODE)


def require_strings(row, fields):
    if set(row) != fields:
        raise ValueError(f"Unexpected/missing fields: {sorted(set(row) ^ fields)}")
    if any(not isinstance(row[k], str) or not row[k].strip() for k in fields):
        raise ValueError("All retrieval input fields must be non-empty strings")


@dataclass(frozen=True)
class Passage:
    passage_id: str
    document_id: str
    text: str
    source_url: str
    source_kind: str
    retrieved_at: str
    acquisition_id: str
    start_char: int
    end_char: int
    text_sha256: str

    def to_dict(self):
        return asdict(self)


def segment(documents, words=180, overlap=30):
    if not isinstance(words, int) or not isinstance(overlap, int) or words <= 0 or not 0 <= overlap < words:
        raise ValueError("Require positive passage words and 0 <= overlap < words")
    passages = []
    for doc in sorted(documents, key=lambda r: r["document_id"]):
        spans = list(re.finditer(r"\S+", doc["text"]))
        for start in range(0, len(spans), words - overlap):
            end = min(start + words, len(spans))
            left, right = spans[start].start(), spans[end - 1].end()
            text = doc["text"][left:right]
            identity = [doc["document_id"], left, right, digest(text)]
            passages.append(Passage(digest(identity), doc["document_id"], text, doc["source_url"],
                                    doc["source_kind"], doc["retrieved_at"], doc["acquisition_id"],
                                    left, right, digest(text)))
            if end == len(spans):
                break
    return passages


def validate_inputs(claims, documents, manifest, claims_hash, corpus_hash, synthetic=False):
    """Attestations are required, but never treated as a complete content audit."""
    expected_dataset = "SYNTHETIC" if synthetic else "AVeriTeC"
    if manifest.get("dataset") != expected_dataset or manifest.get("ontology") != ONTOLOGY:
        raise ValueError("Dataset/official four-class ontology mismatch")
    if manifest.get("synthetic") is not synthetic:
        raise ValueError("Synthetic and research inputs must be explicitly separated")
    if manifest.get("claims_sha256") != claims_hash or manifest.get("corpus_sha256") != corpus_hash:
        raise ValueError("Claims/corpus hash mismatch")
    for flag in ("official_splits_preserved", "corpus_origin_reviewed", "no_gold_labels",
                 "no_gold_justifications", "no_ground_truth_qa", "acquisition_label_free",
                 "answer_leakage_reviewed"):
        if manifest.get(flag) is not True:
            raise ValueError(f"Missing review attestation: {flag}")
    for key in ("corpus_provenance", "claim_id_provenance", "population_status"):
        if not isinstance(manifest.get(key), str) or not manifest[key].strip():
            raise ValueError(f"Missing provenance: {key}")
    if not claims or not documents:
        raise ValueError("Non-empty claims and independent document corpus required")
    seen = set()
    for row in claims:
        require_strings(row, CLAIM_FIELDS)
        if row["sample_id"] in seen:
            raise ValueError("Duplicate sample ID; namespace IDs by official split")
        seen.add(row["sample_id"])
        if row["dataset"] != expected_dataset or row["split"] not in {"train", "dev", "test"}:
            raise ValueError("Official split/dataset mismatch")
    seen = set()
    for row in documents:
        require_strings(row, DOCUMENT_FIELDS)
        if row["document_id"] in seen:
            raise ValueError("Duplicate document ID")
        seen.add(row["document_id"])
        if row["source_kind"] not in {"web_document", "knowledge_base_document"}:
            raise ValueError("Annotated QA/justifications and verdict records are not allowed")
        url = urlsplit(row["source_url"])
        if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password:
            raise ValueError("Public source URL with no credentials required")
    if set(manifest.get("split_counts", {})) != {r["split"] for r in claims}:
        raise ValueError("Split inventory mismatch")
    for split, count in manifest["split_counts"].items():
        if count != sum(r["split"] == split for r in claims):
            raise ValueError("Split count mismatch")


def load_inputs(claims_path, corpus_path, manifest, synthetic=False):
    claims_hash, corpus_hash = file_hash(claims_path), file_hash(corpus_path)
    claims, documents = read_jsonl(claims_path), read_jsonl(corpus_path)
    validate_inputs(claims, documents, manifest, claims_hash, corpus_hash, synthetic)
    return claims, documents


def leakage_scan(claims, documents):
    """Cheap flags, not automatic proof that web content is free of answers."""
    by_text = {}
    for row in claims:
        by_text.setdefault(normalize(row["claim"]), []).append(row["sample_id"])
    overlap = []
    verdict_documents = []
    for doc in documents:
        text = normalize(doc["text"])
        if VERDICT.search(doc["text"]):
            verdict_documents.append(doc["document_id"])
        for claim, ids in by_text.items():
            if claim in text:
                overlap.append({"document_id": doc["document_id"], "sample_ids": ids})
    groups = [ids for ids in by_text.values() if len(ids) > 1]
    cross_split = []
    claim_by_id = {r["sample_id"]: r for r in claims}
    for ids in groups:
        if len({claim_by_id[sid]["split"] for sid in ids}) > 1:
            cross_split.append(ids)
    # Label-blind lexical screening only. Never change official split membership.
    claim_tokens = [set(tokens(r["claim"])) for r in claims]
    near_pairs = []
    near_count = 0
    for i, left in enumerate(claim_tokens):
        if not left:
            continue
        for j in range(i + 1, len(claims)):
            right = claim_tokens[j]
            if not right or min(len(left), len(right)) / max(len(left), len(right)) < .9:
                continue
            similarity = len(left & right) / len(left | right)
            if similarity >= .9:
                near_count += 1
                if len(near_pairs) < 1000:
                    near_pairs.append(dict(left_id=claims[i]["sample_id"], right_id=claims[j]["sample_id"],
                        token_jaccard=similarity, cross_split=claims[i]["split"] != claims[j]["split"]))
    duplicate_docs = {}
    for doc in documents:
        duplicate_docs.setdefault(digest(normalize(doc["text"])), []).append(doc["document_id"])
    source_flags = [d["document_id"] for d in documents if any(marker in d["source_url"].casefold()
        for marker in ("factcheck", "fact-check", "politifact", "snopes", "fullfact", "leadstories"))]
    return dict(verdict_pattern_documents=verdict_documents, claim_text_in_documents=overlap,
                exact_claim_duplicate_groups=groups, cross_split_exact_claim_groups=cross_split,
                near_duplicate_claim_pairs=near_pairs, near_duplicate_pair_count=near_count,
                near_duplicate_pairs_truncated=near_count > len(near_pairs),
                near_duplicate_method="token_set_jaccard>=0.90; flags include numerical/polarity differences",
                exact_document_duplicate_groups=[ids for ids in duplicate_docs.values() if len(ids) > 1],
                fact_check_source_url_flags=source_flags,
                near_duplicate_claim_review="lexical flags require review; not a semantic contamination certificate")
