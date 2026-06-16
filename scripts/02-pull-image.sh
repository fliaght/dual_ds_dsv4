#!/usr/bin/env bash
# Pull the vLLM image on every node in parallel.
# ~9 GB compressed (ARM64).  Concurrent pulls share no bandwidth between nodes.

source "$(dirname "$0")/_lib.sh"

section "Image: $DOCKER_IMAGE on $N_NODES nodes"

all_have_it=true
for node in "${NODE_ARR[@]}"; do
  if ssh_node "$node" "docker images --format '{{.Repository}}:{{.Tag}}' | grep -qx '$DOCKER_IMAGE'"; then
    echo "[$node] already present"
  else
    all_have_it=false
  fi
done

if $all_have_it; then
  echo "✅ Image already on every node — skipping pull."
  echo "Next: bash scripts/03-start-serve.sh"
  exit 0
fi

echo "Pulling (~5 min for the 9 GB ARM64 image, depending on registry bandwidth)..."
for_each_node "docker pull '$DOCKER_IMAGE' 2>&1 | tail -3"

echo "✅ Pull complete.  Next: bash scripts/03-start-serve.sh"
