#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
revision=976aeb152cb184a5095021968bba94da96eb6394
python3 -m venv .venv
if [ ! -d third_party/JaxMARL/.git ]; then
    git clone https://github.com/FLAIROx/JaxMARL.git third_party/JaxMARL
fi
if [ -n "$(git -C third_party/JaxMARL status --porcelain)" ]; then
    echo 'Refusing to change a modified third-party checkout.' >&2
    exit 1
fi
git -C third_party/JaxMARL checkout --detach "$revision"
.venv/bin/python -m pip install -c requirements-lock.txt -e third_party/JaxMARL -e '.[test]'
