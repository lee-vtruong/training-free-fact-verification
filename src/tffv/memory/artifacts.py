"""Provenance envelope for future fold-local rules and demonstrations."""
from tffv.memory import assert_isolated
from tffv.provenance import digest


def knowledge_manifest(memory, evaluation, source_manifest_sha256, generator_config):
    """Caller must supply only the outer-fold memory, before inspecting fold labels.

    This records allowed inputs, not a claim that an opaque generator obeyed them.
    Future generators must be given only these memory records.
    """
    assert_isolated(memory, evaluation)
    if not memory:
        raise ValueError("Empty source memory")
    payload = dict(schema_version=1, source_manifest_sha256=source_manifest_sha256,
                   allowed_sample_ids=sorted(r["sample_id"] for r in memory),
                   excluded_sample_ids=sorted(r["sample_id"] for r in evaluation),
                   memory_content_sha256=digest(memory), generator_config=generator_config,
                   evaluation_gold_used=False)
    return dict(payload, artifact_config_sha256=digest(payload))
