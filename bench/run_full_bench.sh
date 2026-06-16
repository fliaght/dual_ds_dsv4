#!/usr/bin/env bash
# Reproduce the published performance matrix (docs/PERFORMANCE.md).
# Run with the service up.  Single-stream sections assume the interactive-200k
# (MTP-on) recipe; the concurrency section needs the concurrent-32k recipe
# (MAX_NUM_SEQS=4) — restart with RECIPE=concurrent-32k before running part C.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../scripts/_lib.sh"

URL="http://127.0.0.1:$API_PORT/v1/chat/completions"
OUT_DIR="$LOG_DIR/bench-$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_DIR"
echo "Results → $OUT_DIR  (recipe in use: $RECIPE)"
BENCH=("python3" "$SCRIPT_DIR/bench_openai.py" --model "$MODEL" --url "$URL")

echo "── A) Single-stream decode vs output length (ISL=128)"
"${BENCH[@]}" --isl 128 --osl 128 1024 4096 --conc 1 --n 3 \
  --out "$OUT_DIR/A-decode-vs-osl.json"

echo
echo "── B) Single-stream vs input length (long prefill, OSL=256)"
"${BENCH[@]}" --isl 128 4096 16384 --osl 256 --conc 1 --n 2 \
  --out "$OUT_DIR/B-decode-vs-isl.json"

echo
echo "── C) Concurrency sweep (needs RECIPE=concurrent-32k; OSL short)"
if [ "$RECIPE" = "concurrent-32k" ]; then
  "${BENCH[@]}" --isl 128 1024 --osl 128 1024 --conc 1 2 4 --n 8 \
    --out "$OUT_DIR/C-concurrency.json"
else
  echo "   SKIPPED — current recipe is '$RECIPE'.  To run:"
  echo "     1. edit cluster.conf: RECIPE=\"concurrent-32k\""
  echo "     2. bash scripts/04-stop.sh --rm && bash scripts/03-start-serve.sh"
  echo "     3. re-run this script"
fi

echo
echo "✅ Bench complete.  JSONs in $OUT_DIR"
echo "   (For realistic non-degenerate prompts, also try: bash bench/realprompt_sweep.sh)"
