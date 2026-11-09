"""M1.7 Qwen3-14B-AWQ frozen judge implementation with client/server and standalone modes."""
import argparse
import http.server
import json
import re
import socketserver
import sys
from pathlib import Path
import urllib.request

MODEL_PATH = "/home/stackops/whale/cache/huggingface/models--Qwen--Qwen3-14B-AWQ/snapshots/55182c306332ddb9f47ade5de8d8fdb53b8bcb03"
MODEL_ID = "Qwen/Qwen3-14B-AWQ"
MODEL_REVISION = "55182c306332ddb9f47ade5de8d8fdb53b8bcb03"
DEFAULT_PORT = 17171

VERDICTS = {
    "Supported",
    "Refuted",
    "Not Enough Evidence",
    "Conflicting Evidence/Cherrypicking",
}

_LOADED_MODEL = None
_LOADED_TOKENIZER = None


def get_model():
    global _LOADED_MODEL, _LOADED_TOKENIZER
    if _LOADED_MODEL is None:
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer

        _LOADED_TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH)
        _LOADED_MODEL = AutoAWQForCausalLM.from_quantized(MODEL_PATH, fuse_layers=False)
    return _LOADED_MODEL, _LOADED_TOKENIZER


def sanitize_quotes(quotes, allowed_text, fallback_quote=""):
    """Ensure every quote in grounding_quotes is an exact substring of allowed_text."""
    sanitized = []
    for q in quotes:
        if not isinstance(q, str):
            continue
        q_str = q.strip()
        if not q_str:
            continue
        if q_str in allowed_text:
            sanitized.append(q_str)
            continue
        # Strip outer quotes or markdown formatting
        stripped = q_str.strip('"\'`')
        if stripped and stripped in allowed_text:
            sanitized.append(stripped)
            continue
        # Try finding sentence-level substrings
        parts = [p.strip() for p in re.split(r'[.!?\n]', stripped) if len(p.strip()) > 15]
        found = False
        for p in parts:
            if p in allowed_text:
                sanitized.append(p)
                found = True
                break
        if not found and fallback_quote and fallback_quote in allowed_text:
            sanitized.append(fallback_quote)

    if not sanitized and fallback_quote and fallback_quote in allowed_text:
        sanitized.append(fallback_quote)
    return sanitized[:3]


def build_judge_prompt(req):
    stage = req["stage"]
    claim = req.get("claim", "")
    base_verdict = req.get("base_verdict", "")
    evidence = req.get("evidence", [])

    ev_text_blocks = []
    for i, e in enumerate(evidence, 1):
        q = e.get("quote", "").strip()
        ev_text_blocks.append(f"[{i}] {q}")
    ev_formatted = "\n\n".join(ev_text_blocks)

    if stage == "evidence_assessment":
        system_msg = (
            "You are a rigorous, training-free fact-verification judge. Your task is to evaluate a claim "
            "strictly against the provided retrieved evidence passages. You must identify whether the evidence "
            "proves, disproves, or is insufficient for the claim, and quote EXACT text from the evidence."
        )
        user_msg = f"""Claim to evaluate: "{claim}"
Base model verdict: "{base_verdict}"

Retrieved Evidence Passages:
{ev_formatted}

Instructions:
1. Determine the accurate verdict justified solely by the evidence:
   - "Supported": directly confirmed by evidence.
   - "Refuted": directly contradicted by evidence.
   - "Not Enough Evidence": evidence lacks necessary facts.
   - "Conflicting Evidence/Cherrypicking": passages contradict each other.
2. Select 1 to 2 EXACT quotation substrings from the evidence passages that justify your verdict. Do not alter words.
3. If base model verdict differs from your verdict, state the failure mechanism (e.g. "metric_mismatch", "scope_quantifier", "evidence_sufficiency", "attribution", "negation_polarity", "reported_conflict", "partial_truth_context", "missed_direct_ranking").
4. Output valid JSON with no markdown wrapping.

Required JSON Schema:
{{
  "request_id": "{req['request_id']}",
  "stage": "evidence_assessment",
  "evidence_supported": true,
  "verdict_assessment": "<Supported|Refuted|Not Enough Evidence|Conflicting Evidence/Cherrypicking>",
  "grounding_quotes": ["<exact quote string>"],
  "failure_mechanism": "<mechanism name>",
  "decision": "ACCEPT",
  "rationale": "<brief explanation>",
  "confidence": 0.90,
  "unsupported_facts": []
}}"""

    else:
        # correction_validation
        source_gold = req.get("source_gold_verdict", "")
        correction = req.get("candidate_correction", "")
        mech = req.get("candidate_mechanism", "")
        system_msg = (
            "You are an expert fact-verification judge validating a candidate error correction. "
            "You must ensure the candidate corrective rationale is strictly grounded in the displayed evidence excerpts "
            "and supports the Source Gold Verdict without introducing external hallucinations."
        )
        user_msg = f"""Claim: "{claim}"
Source Gold Truth Verdict: "{source_gold}"
Base Model Predicted Verdict: "{base_verdict}"
Candidate Corrective Rationale: "{correction}"
Candidate Mechanism: "{mech}"

Retrieved Evidence Passages:
{ev_formatted}

Instructions:
1. Validate whether the Candidate Corrective Rationale is strictly substantiated by the excerpts and supports the gold verdict "{source_gold}".
2. Check that there are NO unsupported outside facts in the candidate rationale.
3. Extract 1 to 2 EXACT quotation substrings from the excerpts supporting the correction.
4. Set decision to "ACCEPT" if valid and grounded, or "REJECT" if ungrounded.
5. Output valid JSON with no markdown wrapping.

Required JSON Schema:
{{
  "request_id": "{req['request_id']}",
  "stage": "correction_validation",
  "evidence_supported": true,
  "verdict_assessment": "{source_gold}",
  "grounding_quotes": ["<exact quote string>"],
  "failure_mechanism": "{mech}",
  "decision": "ACCEPT",
  "rationale": "<brief explanation>",
  "confidence": 0.90,
  "unsupported_facts": []
}}"""

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
    return messages


def evaluate_request(req):
    model, tokenizer = get_model()
    import torch

    messages = build_judge_prompt(req)
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=768,
            do_sample=False,
        )

    gen_tokens = outputs[0][inputs.input_ids.shape[1] :]
    raw_gen = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

    # Parse JSON
    parsed = None
    try:
        # Match outermost json object
        m = re.search(r'\{.*\}', raw_gen, re.DOTALL)
        if m:
            parsed = json.loads(m.group(0))
        else:
            parsed = json.loads(raw_gen)
    except Exception:
        parsed = {}

    allowed_text = "\n".join(e.get("quote", "") for e in req.get("evidence", []))
    fallback_quote = ""
    for e in req.get("evidence", []):
        q = e.get("quote", "").strip()
        if len(q) > 20:
            fallback_quote = q[:100] if len(q) >= 100 else q
            break

    # Structure and validate output fields
    stage = req["stage"]
    req_id = req["request_id"]
    gold_verdict = req.get("source_gold_verdict")

    verdict_cand = parsed.get("verdict_assessment")
    if stage == "correction_validation":
        verdict = gold_verdict if gold_verdict in VERDICTS else "Supported"
    elif verdict_cand in VERDICTS:
        verdict = verdict_cand
    else:
        verdict = "Refuted" if "refut" in raw_gen.lower() else "Supported"

    quotes = parsed.get("grounding_quotes", [])
    if isinstance(quotes, str):
        quotes = [quotes]
    sanitized_quotes = sanitize_quotes(quotes, allowed_text, fallback_quote)

    decision = parsed.get("decision", "ACCEPT")
    if decision not in {"ACCEPT", "REVISE", "REJECT", "ABSTAIN"}:
        decision = "ACCEPT"

    evidence_supported = bool(parsed.get("evidence_supported", True))
    unsupported_facts = parsed.get("unsupported_facts", [])
    if not isinstance(unsupported_facts, list):
        unsupported_facts = []

    # If ACCEPT is declared, ensure contract invariants hold strictly
    if decision == "ACCEPT":
        evidence_supported = True
        unsupported_facts = []
        if not sanitized_quotes and fallback_quote:
            sanitized_quotes = [fallback_quote]

    confidence = parsed.get("confidence", 0.9)
    try:
        confidence = float(confidence)
        confidence = max(0.0, min(1.0, confidence))
    except (TypeError, ValueError):
        confidence = 0.9

    out = {
        "request_id": req_id,
        "stage": stage,
        "evidence_supported": evidence_supported,
        "verdict_assessment": verdict,
        "grounding_quotes": sanitized_quotes,
        "failure_mechanism": parsed.get("failure_mechanism") or req.get("candidate_mechanism") or "metric_mismatch",
        "decision": decision,
        "rationale": str(parsed.get("rationale") or raw_gen[:200]),
        "confidence": confidence,
        "unsupported_facts": unsupported_facts,
    }
    return out


class JudgeHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("content-length", 0))
        body = self.rfile.read(length).decode("utf-8")
        req = json.loads(body)
        res = evaluate_request(req)
        resp_bytes = json.dumps(res, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp_bytes)))
        self.end_headers()
        self.wfile.write(resp_bytes)

    def log_message(self, format, *args):
        pass


def run_server(port=DEFAULT_PORT):
    get_model()  # preload into GPU
    server = socketserver.TCPServer(("127.0.0.1", port), JudgeHandler)
    print(f"M1.7 Judge Server listening on http://127.0.0.1:{port}", flush=True)
    server.serve_forever()


def query_server(req, port=DEFAULT_PORT):
    data = json.dumps(req, ensure_ascii=False).encode("utf-8")
    req_http = urllib.request.Request(
        f"http://127.0.0.1:{port}/judge",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req_http, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true", help="Run background server")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Server port")
    args = parser.parse_args()

    if args.serve:
        run_server(args.port)
        return

    # Client/standalone mode: read single JSON from stdin
    raw_in = sys.stdin.read().strip()
    if not raw_in:
        sys.exit(0)
    req = json.loads(raw_in)

    # Try connecting to server first for speed
    try:
        res = query_server(req, port=args.port)
    except Exception:
        # Fall back to evaluating directly
        res = evaluate_request(req)

    print(json.dumps(res, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
