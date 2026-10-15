#!/usr/bin/env bash
# Smoke test the vLLM OpenAI endpoint.  Run after the service is up.
# Sends a handful of prompts (en/zh/ja + math + code) and asserts the math one
# returns "391".  If English is garbled but Chinese is coherent, the Marlin→
# DeepGEMM layer-42 workaround didn't take effect (see Docs/kedb/runtime.md).

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../scripts/_lib.sh"

URL="${URL:-http://127.0.0.1:$API_PORT/v1/chat/completions}"
TEMPLATE_ARGS=""
if [ "$RUNTIME" = v41 ]; then
  TEMPLATE_ARGS=',"chat_template_kwargs":{"enable_thinking":false}'
fi

PROMPTS=(
  "What is 17 * 23? Reply with just the number."
  "用一句话介绍 DeepSeek-V4-Flash。"
  "一語で答えて：フランスの首都はどこですか？"
  "Write one line of Python that sums 1..100."
  "Capital of France in one word."
)

ok=0
FIRST=""
for i in "${!PROMPTS[@]}"; do
  Q="${PROMPTS[$i]}"
  echo "── smoke #$((i+1)): $Q"
  if ! RESP=$(curl -fsS --max-time 120 "$URL" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"$Q\"}],\"max_tokens\":64,\"temperature\":0.0$TEMPLATE_ARGS}"); then
    echo "  ❌ HTTP/connection failure"
    continue
  fi
  CONTENT=$(jq -er '.choices[0].message.content | select(type == "string" and length > 0)' <<< "$RESP") || CONTENT=""
  TOK=$(jq -r '.usage.completion_tokens // 0' <<< "$RESP") || TOK=0
  if [ "$i" = 0 ]; then FIRST="$CONTENT"; fi
  if [ -n "$CONTENT" ]; then
    echo "  ✅ ${TOK} tok: ${CONTENT:0:140}"
    ok=$((ok + 1))
  else
    echo "  ❌ failed: $CONTENT"
  fi
  echo
done

# Reuse the first response: no duplicate inference just to check the answer.
if grep -Eq '(^|[^0-9])391([^0-9]|$)' <<< "$FIRST" && [ "$ok" -eq 5 ]; then
  echo "✅ PASS: $ok/5 prompts answered, math returned 391."
  exit 0
else
  echo "❌ FAIL: $ok/5 answered; math='$FIRST' (expected to contain 391)."
  exit 1
fi
