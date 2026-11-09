from pathlib import Path
from tffv.experiments.m19g_guardrails import verify_knowledge_store

def main():
    claim_ids = [f"dev:{i}" for i in range(500)]
    ks_dir = "data_store/knowledge_store/dev/uncompressed"
    print(f"Verifying {len(claim_ids)} claims against {ks_dir}...")
    stats = verify_knowledge_store(claim_ids, ks_dir)
    print("Verification SUCCESSFUL!")
    print(f"Total claims checked: {stats['total_claims_checked']}")
    print(f"Total valid files: {stats['valid_files']}")
    print(f"Total documents: {stats['total_documents']:,}")
    print(f"Total characters: {stats['total_characters']:,}")

if __name__ == "__main__":
    main()
