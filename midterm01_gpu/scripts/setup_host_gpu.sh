#!/usr/bin/env bash
# Run explicitly on the Ubuntu host with sudo. Never restarts Docker.
set -euo pipefail
if (( EUID != 0 )); then
    printf 'Run on the Ubuntu host: sudo bash %q\n' "$0" >&2
    exit 1
fi
for command in curl gpg docker nvidia-smi python3; do
    command -v "$command" >/dev/null || { printf 'Missing host command: %s\n' "$command" >&2; exit 1; }
done
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
if ! docker info --format '{{json .CDISpecDirs}}' | python3 -c \
    'import json,sys; paths=json.load(sys.stdin); sys.exit(0 if "/var/run/cdi" in (paths or []) else 1)'; then
    printf 'Docker native CDI with /var/run/cdi is required; no daemon configuration was changed.\n' >&2
    exit 1
fi

task_tmp_dir="$(mktemp -d)"
trap 'rm -rf -- "$task_tmp_dir"' EXIT
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey -o "$task_tmp_dir/nvidia.asc"
gpg --batch --yes --dearmor --output "$task_tmp_dir/nvidia.gpg" "$task_tmp_dir/nvidia.asc"
install -m 0644 "$task_tmp_dir/nvidia.gpg" /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    > "$task_tmp_dir/nvidia-container-toolkit.list"
install -m 0644 "$task_tmp_dir/nvidia-container-toolkit.list" /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update
# CDI needs only the base package; avoid changing the default Docker runtime.
apt-get install -y --no-install-recommends nvidia-container-toolkit-base
install -d -m 0755 /var/run/cdi
nvidia-ctk cdi generate --output=/var/run/cdi/nvidia.yaml
nvidia-ctk cdi list
printf '\nHost CDI setup complete. Docker was not restarted.\n'
printf 'Next: bash scripts/dev.sh up && bash scripts/dev.sh cuda-check\n'

