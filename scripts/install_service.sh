#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
binary="$root_dir/build/pi/edge_runner"

if [[ ! -x "$binary" ]]; then
  echo "Missing $binary. Run bash scripts/build_pi.sh first."
  exit 1
fi

run_user="${SUDO_USER:-$USER}"
run_group="$(id -gn "$run_user")"

sudo install -d -m 0755 /opt/edge-runner/bin /opt/edge-runner/frontend
sudo install -d -o "$run_user" -g "$run_group" -m 0750 /var/lib/edge-runner/models
sudo install -m 0755 "$binary" /opt/edge-runner/bin/edge_runner
sudo cp -a "$root_dir/frontend/dist/." /opt/edge-runner/frontend/
sed -e "s/__RUN_USER__/$run_user/g" -e "s/__RUN_GROUP__/$run_group/g" "$root_dir/deploy/edge-runner.service" | sudo tee /etc/systemd/system/edge-runner.service >/dev/null
sudo systemctl daemon-reload
sudo systemctl enable --now edge-runner.service

echo "Edge Runner is installed and running."
echo "Open http://$(hostname -I | awk '{print $1}'):8080 from another device on this network."
echo "Logs: journalctl -u edge-runner -f"

