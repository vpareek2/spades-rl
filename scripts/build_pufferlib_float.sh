#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PUFFER="$ROOT/PufferLib"

CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$HOME/.local/bin:$PATH"

CUDNN_LIB="$ROOT/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib"
NCCL_LIB="$ROOT/.venv/lib/python3.12/site-packages/nvidia/nccl/lib"
if [[ -d "$CUDNN_LIB" ]]; then
  ln -sf libcudnn.so.9 "$CUDNN_LIB/libcudnn.so"
fi
if [[ -d "$NCCL_LIB" ]]; then
  ln -sf libnccl.so.2 "$NCCL_LIB/libnccl.so"
fi
if [[ -e /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1 && ! -e /usr/lib/x86_64-linux-gnu/libnvidia-ml.so ]]; then
  sudo ln -sf /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1 /usr/lib/x86_64-linux-gnu/libnvidia-ml.so
fi

export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${CUDNN_LIB:-}:${NCCL_LIB:-}:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:--ccbin /usr/bin/g++-11}"

python3 - <<'PY'
from pathlib import Path

path = Path("PufferLib/src/kernels.cu")
text = path.read_text()
old = """inline void cast_dispatch(precision_t* dst, const float* src, int n, cudaStream_t stream) {
    cast<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(dst, src, n);
}

#ifndef PRECISION_FLOAT
"""
new = """#ifndef PRECISION_FLOAT
inline void cast_dispatch(precision_t* dst, const float* src, int n, cudaStream_t stream) {
    cast<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(dst, src, n);
}

"""
if old in text:
    path.write_text(text.replace(old, new))
elif new not in text:
    raise SystemExit("PufferLib cast_dispatch block not found; inspect PufferLib/src/kernels.cu")
PY

cd "$PUFFER"
CC="${CC:-clang}" CXX="${CXX:-/usr/bin/g++-11}" \
  uv run --with pybind11 --with rich_argparse ./build.sh breakout --float

cd "$ROOT"
uv run python - <<'PY'
from pufferlib import _C

print(_C.__file__)
print("precision_bytes", _C.precision_bytes)
PY
