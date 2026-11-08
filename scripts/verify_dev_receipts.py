from pathlib import Path
from tffv.experiments.m19g_guardrails import verify_retrieval_receipts

def main():
    claim_ids = [f"dev:{i}" for i in range(500)]
    receipts_path = "outputs/m19g-evidence-complete-v1/dev_retrieval_receipts.jsonl"
    print(f"Verifying {len(claim_ids)} claims against {receipts_path}...")
    stats = verify_retrieval_receipts(claim_ids, receipts_path, require_all_claims=True)
    print("Verification SUCCESSFUL!")
    print(f"Status: {stats['status']}")
    print(f"Total receipts: {stats['total_receipts']}")
    print(f"Total passages: {stats['total_passages']}")
    print(f"Total characters: {stats['total_chars']:,}")

if __name__ == "__main__":
    main()
