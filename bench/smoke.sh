#!/usr/bin/env bash
# Smoke test the vLLM OpenAI endpoint.  Run after the service is up.
# Sends a handful of prompts (en/zh/ja + math + code) and asserts the math one
# returns "391".  If English is garbled but Chinese is coherent, the Marlin→
# DeepGEMM layer-42 workaround didn't take effect (see docs/TROUBLESHOOTING.md).

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../scripts/_lib.sh"

URL="${URL:-http://127.0.0.1:$API_PORT/v1/chat/completions}"

PROMPTS=(
  "What is 17 * 23? Reply with just the number."
  "用一句话介绍 DeepSeek-V4-Flash。"
  "一語で答えて：フランスの首都はどこですか？"
  "Write one line of Python that sums 1..100."
  "Capital of France in one word."
)

ok=0
for i in "${!PROMPTS[@]}"; do
  Q="${PROMPTS[$i]}"
  echo "── smoke #$((i+1)): $Q"
  RESP=$(curl -sS --max-time 120 "$URL" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"$Q\"}],\"max_tokens\":64,\"temperature\":0.0}")
  CONTENT=$(echo "$RESP" | jq -r '.choices[0].message.content // .error.message // "<empty>"')
  TOK=$(echo "$RESP" | jq -r '.usage.completion_tokens // 0')
  if [ -n "$CONTENT" ] && [ "$CONTENT" != "<empty>" ] && [ "$CONTENT" != "null" ]; then
    echo "  ✅ ${TOK} tok: ${CONTENT:0:140}"
    ok=$((ok + 1))
    [ "$i" = "0" ] && echo "$CONTENT" | grep -q "391" && echo "     (math check: 391 ✓)"
  else
    echo "  ❌ failed: $CONTENT"
  fi
  echo
done

# The first prompt is the hard assertion.
FIRST=$(curl -sS --max-time 120 "$URL" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 17 * 23? Reply with just the number.\"}],\"max_tokens\":32,\"temperature\":0.0}" \
  | jq -r '.choices[0].message.content // ""')

if echo "$FIRST" | grep -q "391" && [ "$ok" -ge 4 ]; then
  echo "✅ PASS: $ok/5 prompts answered, math returned 391."
  exit 0
else
  echo "❌ FAIL: $ok/5 answered; math='$FIRST' (expected to contain 391)."
  exit 1
fi
