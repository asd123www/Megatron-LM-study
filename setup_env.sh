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

as_root pip uninstall -y flash-attn-3 flash_attn_3 flash-attn-hopper
MAX_JOBS=64 python -m pip -v install --no-build-isolation "transformer-engine[pytorch]==2.10.0"
