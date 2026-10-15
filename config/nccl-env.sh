# NCCL / transport environment for vLLM cross-node TP on DGX Spark's RoCE fabric.
# Sourced by scripts/run-node.sh, which passes each var into the container with
# `docker run -e`.  Single source of truth for the "conservative NCCL" knobs.
#
# Rationale and scope: see Docs/kedb/runtime.md and Knowledge/architecture.md.

# --- Bind every *socket* transport (bootstrap, Gloo, fallback) to $IFACES from cluster.conf ---
# That is the RoCE netdevs on a classic setup, or the management NIC when the RoCE ports carry
# no IPs (then set NCCL_IB_GID_INDEX below).  NCCL's RDMA data path does NOT depend on this:
# it uses every active RoCE HCA on the node regardless of IFACES.
# Set by run-node.sh just before sourcing this file; re-exported here for clarity.
: "${IFACES:?IFACES must be set before sourcing nccl-env.sh}"
export NCCL_SOCKET_IFNAME="$IFACES"
export GLOO_SOCKET_IFNAME="$IFACES"
export TP_SOCKET_IFNAME="$IFACES"
export UCX_NET_DEVICES="$IFACES"
export OMPI_MCA_btl_tcp_if_include="$IFACES"

# --- Disable advanced RDMA features that this image's NCCL tries by default but
# GB10 / DGX Spark RoCE does NOT support.  Without these, rank 1 segfaults during
# inter-node TP setup in transport/net_ib/gdaki/gin_host_gdaki.cc (exit 139). ---
export NCCL_GIN_ENABLE=0        # GPU Initiated Networking
export NCCL_WIN_ENABLE=0        # NCCL windows
export NCCL_DMABUF_ENABLE=0     # DMA-BUF buffer registration
export NCCL_GRAPH_REGISTER=0    # CUDA-graph buffer registration
export NCCL_LOCAL_REGISTER=0    # local buffer pre-registration
export NCCL_MNNVL_ENABLE=0      # Multi-Node NVLink (not present on Spark)
export NCCL_IB_MERGE_NICS=0     # keep the 2 IB ports as separate NICs

# --- Debug. INFO lets you confirm RDMA: post-boot the log must show
# `via NET/IB/0` + `via NET/IB/1` and ZERO `via NET/Socket` / `gdaki` / `MNNVL`.
# scripts/03-start-serve.sh greps for exactly this. ---
export NCCL_DEBUG=INFO

# Ordered list of every NCCL/transport var above, so callers can iterate to build
# `-e VAR=val` flags without re-listing them.
NCCL_ENV_VARS="NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME TP_SOCKET_IFNAME UCX_NET_DEVICES \
OMPI_MCA_btl_tcp_if_include NCCL_GIN_ENABLE NCCL_WIN_ENABLE NCCL_DMABUF_ENABLE \
NCCL_GRAPH_REGISTER NCCL_LOCAL_REGISTER NCCL_MNNVL_ENABLE NCCL_IB_MERGE_NICS NCCL_DEBUG"

# --- Which RoCE GID NCCL uses (optional; NCCL_IB_GID_INDEX in cluster.conf). ---
# Unset  -> NCCL auto-selects, which needs an IPv4 address on each RoCE port (classic setup).
# 0      -> RoCE v1 link-local GID derived from the port MAC: pure layer-2, works over the
#           direct QSFP cable with NO IPs on the RoCE ports, regardless of which QSFP port
#           each end is plugged into.  Measured 19.5 GB/s all-reduce on this pair.
if [ -n "${NCCL_IB_GID_INDEX:-}" ]; then
  export NCCL_IB_GID_INDEX
  NCCL_ENV_VARS="$NCCL_ENV_VARS NCCL_IB_GID_INDEX"
fi
