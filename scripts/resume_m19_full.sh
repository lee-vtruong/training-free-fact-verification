#!/usr/bin/env bash
# Resumption launcher for Milestone M1.9 Evaluation
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Resuming M1.9 evaluation..."
bash "$PROJECT_ROOT/scripts/launch_m19_full.sh" "$@"
