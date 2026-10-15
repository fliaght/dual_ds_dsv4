# Configuration only: no SSH, Docker, directory creation, or service actions.
# Source this file, then call load_cluster_config [required|optional].

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

load_cluster_config() {
  local mode="${1:-required}"
  if [ -f "$ROOT_DIR/cluster.conf" ]; then
    # Trusted, site-owned shell configuration; never commit it.
    # shellcheck source=/dev/null
    source "$ROOT_DIR/cluster.conf"
  elif [ "$mode" != optional ]; then
    echo "ERROR: $ROOT_DIR/cluster.conf missing. Copy from cluster.conf.example and edit." >&2
    return 1
  fi

  RECIPE="${RECIPE:-interactive-128k}"
  if [[ ! "$RECIPE" =~ ^[a-zA-Z0-9_-]+$ ]]; then
    echo "ERROR: invalid recipe name: $RECIPE" >&2
    return 1
  fi
  RECIPE_FILE="$ROOT_DIR/config/recipes/${RECIPE}.env"
  if [ ! -f "$RECIPE_FILE" ]; then
    echo "ERROR: recipe '$RECIPE' -> $RECIPE_FILE not found." >&2
    return 1
  fi
  # shellcheck source=/dev/null
  RUNTIME=v4
  source "$RECIPE_FILE"
  if [ "$RUNTIME" = v41 ]; then
    # V4.1 runtime knobs; cluster.conf may override. Both ranks must agree (v41-service.sh).
    V41_CUDAGRAPH_SIZES="${V41_CUDAGRAPH_SIZES:-1 2 3 4 6 8 12 18 24}"
    V41_PREFIX_RETENTION_INTERVAL="${V41_PREFIX_RETENTION_INTERVAL:-0}"
    V41_EMPTY_CACHE_MEMAVAIL_GIB="${V41_EMPTY_CACHE_MEMAVAIL_GIB:-12}"
    V41_MALLOC_ARENA_MAX="${V41_MALLOC_ARENA_MAX:-}"
    V41_PROFILER_DIR="${V41_PROFILER_DIR:-}"   # maintenance only: adds /start_profile
  fi
}
