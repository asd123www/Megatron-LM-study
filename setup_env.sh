#!/usr/bin/env bash
set -euo pipefail

# Refuse to be run with sudo to avoid breaking git/SSH/home paths
if [[ $EUID -eq 0 ]]; then
  echo "Please run this script without sudo."
  exit 1
fi

# Ask for sudo once and keep it alive
sudo -v
( while true; do sleep 60; sudo -n true || exit; done ) &
SUDO_KEEPALIVE_PID=$!
trap 'kill "$SUDO_KEEPALIVE_PID"' EXIT

as_root() {
  # We know we're not root here, so just sudo the command
  sudo "$@"
}


# --------------- actual commands -----------------
export http_proxy=http://sys-proxy-rd-relay.byted.org:8118  https_proxy=http://sys-proxy-rd-relay.byted.org:8118  no_proxy=code.byted.org

rm -rf /opt/tiger/mariana/megatron
# `ModuleNotFoundError: No module named 'flash_attn_3.flash_attn_interface'` error...
as_root pip uninstall -y flash-attn-3 flash_attn_3 flash-attn-hopper


# torch 2.7.1
# zezhou: you need 4.56.2 for gpt-oss.
pip install transformers==4.56.2
pip install bitsandbytes==0.45.5
pip install accelerate
MAX_JOBS=64 python -m pip -v install flash-attn --no-build-isolation

# Megatron-fsdp
# check `https://pypi.org/project/megatron-fsdp/0.1.0rc3/#description`
MAX_JOBS=64 python -m pip -v install --no-build-isolation "transformer-engine[pytorch]==2.10.0"
pip install "megatron-core[mlm]"
MAX_JOBS=64 python -m pip -v install "git+https://github.com/NVIDIA/Megatron-LM.git@4cf968cd26cc0e8cfcb65eebac6e3a60b220699d"
pip install megatron-fsdp

# python -c "import megatron.core; import pkgutil; print('mcore file:', megatron.core.__file__); print('mcore submodules:', [m.name for m in pkgutil.iter_modules(megatron.core.__path__)])"
python -m pip install --no-cache-dir "triton==3.3.0"

pip install transformers==4.56.2
