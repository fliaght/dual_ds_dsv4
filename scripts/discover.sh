#!/usr/bin/env bash
# Discover this Spark pair's RoCE NIC names + IPs and print ready-to-paste
# cluster.conf values.  Use it BEFORE writing cluster.conf on a fresh pair, so
# you never hand-type the casing-sensitive NIC names (e.g. enP2p1s0f0np0) or
# accidentally use a management IP for NODES.
#
#   bash scripts/discover.sh                       # inspect THIS node only (the HEAD)
#   bash scripts/discover.sh <worker-ssh-addr> ...  # also SSH each worker and inspect it
#
# The host you run it on is treated as the HEAD (first in NODES).  Worker
# arguments are SSH addresses (any IP that reaches the box — even a management
# IP); the suggested NODES uses the discovered *RoCE* IP regardless.
#
# Standalone: does NOT require cluster.conf to exist yet.

set -uo pipefail

# Remote-safe snippet: list Up RoCE netdevs and each one's first IPv4.
RDISCOVER='
for nic in $(ibdev2netdev 2>/dev/null | awk "/Up\)/{print \$5}" | tr -d "()"); do
  ip4=$(ip -4 -br addr show dev "$nic" 2>/dev/null | awk "{print \$3}" | head -1 | cut -d/ -f1)
  printf "%s %s\n" "$nic" "${ip4:-(none)}"
done'

gather() {  # $1 = "" for local, else ssh address
  local host="$1"
  if [ -z "$host" ]; then
    bash -c "$RDISCOVER"
  else
    ssh -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new "$host" "$RDISCOVER"
  fi
}

HOSTS=( "" "$@" )                 # "" = local (HEAD); args = worker ssh addrs
declare -a NICS IP1 IP2
warn=0

echo "Discovering RoCE fabric across ${#HOSTS[@]} host(s)..."
echo
idx=0
for h in "${HOSTS[@]}"; do
  label="${h:-"(this node / HEAD)"}"
  echo "── $label"
  if ! out=$(gather "$h") || [ -z "$out" ]; then
    echo "    ERROR: discovery failed (is ibdev2netdev installed and a RoCE NIC Up?)"
    warn=1; idx=$((idx+1)); continue
  fi
  echo "$out" | awk '{printf "    %-16s %s\n", $1, $2}'
  NICS[$idx]=$(echo "$out" | awk '{print $1}' | paste -sd, -)
  IP1[$idx]=$(echo "$out" | awk 'NR==1{print $2}')
  IP2[$idx]=$(echo "$out" | awk 'NR==2{print $2}')
  echo "$out" | grep -q '(none)' && { echo "    ⚠ a RoCE NIC has no IPv4 — assign one before serving."; warn=1; }
  idx=$((idx+1))
done

echo
echo "Suggested cluster.conf values:"
# NODES = each host's first-RoCE-NIC IP, head first.
nodes=""; for v in "${IP1[@]}"; do nodes+="${v} "; done
echo "  NODES=\"$(echo "$nodes" | sed 's/ *$//')\""
echo "  IFACES=\"${NICS[0]:-<none>}\""
# TRANSFER_PEER = first worker's second-RoCE-NIC IP (201.x), for the weight rsync.
if [ "${#HOSTS[@]}" -gt 1 ] && [ -n "${IP2[1]:-}" ] && [ "${IP2[1]}" != "(none)" ]; then
  echo "  TRANSFER_PEER=\"${IP2[1]}\""
elif [ -n "${IP2[0]:-}" ] && [ "${IP2[0]}" != "(none)" ]; then
  echo "  # TRANSFER_PEER = worker's 2nd-port (201.x) IP — run with the worker arg to fill it"
fi

# Consistency: do NIC name sets match across hosts?
mismatch=0
for n in "${NICS[@]:1}"; do [ -n "$n" ] && [ "$n" != "${NICS[0]}" ] && mismatch=1; done
echo
if [ "$mismatch" -eq 1 ]; then
  echo "⚠ RoCE NIC names DIFFER across hosts — this stack binds one IFACES list to"
  echo "  both ranks, so names must match.  Check NIC enumeration on each box."
  warn=1
elif [ "${#HOSTS[@]}" -gt 1 ]; then
  echo "✅ NIC name sets match across all hosts."
else
  echo "ℹ Inspected this node only.  Re-run as: bash scripts/discover.sh <worker-ssh-addr>"
fi
echo
echo "Next: cp cluster.conf.example cluster.conf, paste the values above, then"
echo "      bash scripts/00-precheck.sh   (it re-validates NODES IPs are on IFACES)."
exit "$warn"
