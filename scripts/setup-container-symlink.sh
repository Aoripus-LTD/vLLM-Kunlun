#!/bin/bash
# scripts/setup-container-symlink.sh
#
# Make vllm serve inside the vllm-kunlun container load the
# /home/vLLM-Kunlun-Aoripus fork (which the user edits and commits)
# instead of the pre-installed /opt/vllm_kunlun snapshot.
#
# Idempotent: safe to re-run. The pre-installed package is backed
# up as vllm_kunlun.bak on first run.
#
# Run from host:
#   docker exec vllm-kunlun bash /workspace/VLLM-Kunlun-Aoripus/scripts/setup-container-symlink.sh

set -euo pipefail

SITE_PKG="/opt/vllm_kunlun/lib/python3.10/site-packages"
FORK_SRC="/workspace/vLLM-Kunlun-Aoripus/vllm_kunlun"
TARGET="${SITE_PKG}/vllm_kunlun"
BAK="${SITE_PKG}/vllm_kunlun.bak"
ORIG="${SITE_PKG}/vllm_kunlun.orig"

# 1. Validate fork source exists with correct case
if [ ! -d "${FORK_SRC}" ]; then
  echo "FATAL: fork source not found: ${FORK_SRC}" >&2
  echo "  (note: case-sensitive — must be lowercase 'v' in vLLM)" >&2
  exit 1
fi
if [ ! -f "${FORK_SRC}/__init__.py" ]; then
  echo "FATAL: ${FORK_SRC}/__init__.py missing — fork is incomplete" >&2
  exit 1
fi

# 2. If target is already a correct symlink, nothing to do
if [ -L "${TARGET}" ] && [ "$(readlink "${TARGET}")" = "${FORK_SRC}" ]; then
  echo "Symlink already correct: ${TARGET} -> ${FORK_SRC}"
else
  # 3. Move aside whatever's there (real dir or wrong symlink)
  if [ -e "${TARGET}" ] || [ -L "${TARGET}" ]; then
    rm -rf "${BAK}" 2>/dev/null || true
    if [ -e "${ORIG}" ]; then
      mv "${ORIG}" "${BAK}"
    else
      mv "${TARGET}" "${BAK}"
    fi
    echo "Backed up existing ${TARGET} -> ${BAK}"
  fi

  # 4. Create symlink
  ln -s "${FORK_SRC}" "${TARGET}"
  echo "Created symlink: ${TARGET} -> ${FORK_SRC}"
fi

# 5. Clear stale .pyc caches that may shadow the new source
find "${SITE_PKG}" -maxdepth 2 -name "__pycache__" -type d \
  -path "*/vllm_kunlun/*" -exec rm -rf {} + 2>/dev/null || true
echo "Cleared .pyc cache for vllm_kunlun"

# 6. Verify import works and points to fork
VERIFY=$(python -W ignore -c "
import vllm_kunlun, inspect
from vllm.config.model import ModelConfig
src = inspect.getsource(ModelConfig.is_deepseek_mla.fget)
print('OK' if 'chatglm' in src else 'NO_CHATGLM')
" 2>&1 | tail -1)
if [ "${VERIFY}" = "OK" ]; then
  echo "VERIFY: fork is active (chatglm in whitelist)"
else
  echo "VERIFY FAILED: ${VERIFY}" >&2
  exit 2
fi

echo "Done. vllm serve in this container now uses the fork."
