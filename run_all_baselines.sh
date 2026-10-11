#!/usr/bin/env bash
set -euo pipefail
cora_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -x "$cora_root/.venv/bin/python" ]]; then
    echo 'Missing Python environment. Run bash scripts/setup.sh once.' >&2
    exit 1
fi
cd "$cora_root"
export PYTHONDONTWRITEBYTECODE=1
exec "$cora_root/.venv/bin/python" -u -m cora.benchmarks.campaign "$@"
