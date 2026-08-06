#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
nvrtc_source="$(find "$repo_root/.venv" -path '*/nvidia/cuda_nvrtc/lib/libnvrtc.so.*' -type f -print -quit 2>/dev/null || true)"

if [[ -z "$nvrtc_source" ]]; then
    echo "NVRTC library not found. Run 'uv sync' first." >&2
    exit 1
fi

nvrtc_dir="${TMPDIR:-/tmp}/fnoreg-cuda-libs"
mkdir -p "$nvrtc_dir"
ln -sfn "$nvrtc_source" "$nvrtc_dir/libnvrtc.so"

export LD_LIBRARY_PATH="$nvrtc_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${TMPDIR:-/tmp}/uv-cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/matplotlib}"

cd "$repo_root"
exec uv run "$@"
