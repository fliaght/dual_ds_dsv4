#!/usr/bin/env bash
# Single-stream decode tok/s on REALISTIC prompts (a public-domain story snippet),
# not the degenerate "benchmark benchmark…" filler that artificially suppresses
# MTP acceptance.  Builds prompts of ~N input tokens, asks for OSL continuation,
# reports prompt/completion tokens, wall time, and tok/s per ISL.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../scripts/_lib.sh"

URL="${URL:-http://127.0.0.1:$API_PORT/v1/chat/completions}"
OSL="${OSL:-256}"
RESULTS_JSON="${RESULTS_JSON:-$LOG_DIR/realprompt-$(date +%Y%m%d_%H%M%S).json}"

SEED='Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do: once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, "and what is the use of a book," thought Alice "without pictures or conversations?" So she was considering in her own mind (as well as she could, for the hot day made her feel very sleepy and stupid), whether the pleasure of making a daisy-chain would be worth the trouble of getting up and picking the daisies, when suddenly a White Rabbit with pink eyes ran close by her. '

build_prompt() {            # ~4 chars/token
  local target_chars=$(( $1 * 4 )) prompt=""
  while [ "${#prompt}" -lt "$target_chars" ]; do prompt="${prompt}${SEED}"; done
  echo "${prompt:0:$target_chars}"
}

echo "[" > "$RESULTS_JSON"; first=1
for ISL_TARGET in 128 4096 16384 65536 100000; do
  echo ">>> ISL ~= ${ISL_TARGET}  OSL=${OSL}"
  prompt=$(build_prompt "$ISL_TARGET")
  payload=$(mktemp)
  jq -n --arg model "$MODEL" --rawfile p <(printf '%s' "$prompt") --argjson n "$OSL" \
    '{model:$model, messages:[{role:"user", content:$p}], max_tokens:$n, temperature:0.0, ignore_eos:true}' \
    > "$payload"
  t0=$(date +%s.%N)
  resp=$(curl -sS --max-time 1800 "$URL" -H 'Content-Type: application/json' --data-binary @"$payload")
  t1=$(date +%s.%N)
  rm -f "$payload"
  wall=$(echo "$t1 - $t0" | bc -l)
  ptoks=$(echo "$resp" | jq -r '.usage.prompt_tokens // 0')
  ctoks=$(echo "$resp" | jq -r '.usage.completion_tokens // 0')
  fr=$(echo "$resp" | jq -r '.choices[0].finish_reason // "?"')
  tps=$(echo "scale=2; $ctoks / $wall" | bc -l)
  printf "  ptoks=%s ctoks=%s wall=%.1fs fr=%s tok/s=%s\n" "$ptoks" "$ctoks" "$wall" "$fr" "$tps"
  [ $first -eq 0 ] && echo "," >> "$RESULTS_JSON"; first=0
  echo "{\"isl_target\":${ISL_TARGET},\"ptoks\":${ptoks},\"ctoks\":${ctoks},\"wall_s\":${wall},\"tok_per_s\":${tps},\"finish_reason\":\"${fr}\"}" >> "$RESULTS_JSON"
done
echo "]" >> "$RESULTS_JSON"
echo "wrote $RESULTS_JSON"
