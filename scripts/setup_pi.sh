#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "This setup script must run on Raspberry Pi OS or another Debian-based Linux system."
  exit 1
fi

arch="$(uname -m)"
if [[ "$arch" != "aarch64" ]]; then
  echo "Warning: detected $arch. Raspberry Pi OS 64-bit (aarch64) is strongly recommended for Pi 4 inference."
fi

sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  build-essential ca-certificates cmake git libomp-dev ninja-build nodejs npm

node_major="$(node -p 'Number(process.versions.node.split(".")[0])')"
if (( node_major < 20 )); then
  echo "Node.js 20 or newer is required to build the React assets. Build frontend/dist on another PC, or install a current Node.js release."
  exit 1
fi

echo "Pi build prerequisites are ready."
echo "Next: bash scripts/build_pi.sh"

