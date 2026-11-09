#!/usr/bin/env bash
# Resumes an interrupted Milestone M1.9f evaluation run
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Resuming M1.9f Evaluation using launch_m19f_clean.sh..."
bash "$PROJECT_ROOT/scripts/launch_m19f_clean.sh" "$@"
