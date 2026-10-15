#!/usr/bin/env bash
# Zero-to-serving one-shot.  Run from the HEAD node after editing cluster.conf.
#
# Equivalent to:
#   bash scripts/00-precheck.sh
#   bash scripts/02-pull-image.sh
#   bash scripts/03-start-serve.sh
#
# (01-download-weights.sh is NOT included — weights are ~158 GB and you only
#  fetch them once; run it manually the first time, then 00-precheck confirms
#  they're present.)  Each step is idempotent; re-run any in isolation.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_config.sh"
load_cluster_config

for s in 00-precheck.sh 02-pull-image.sh 03-start-serve.sh; do
  echo
  echo "▶▶▶ $s"
  bash "$SCRIPT_DIR/$s"
done

echo
echo "🎉 Service up at http://127.0.0.1:${API_PORT}"
echo "Next: bash $SCRIPT_DIR/../bench/smoke.sh"
