#!/usr/bin/env bash
set -euo pipefail

cd /workspace

missing_packages="$(
python3 - <<'PY'
missing = []
for module_name in ("openai", "rich"):
    try:
        __import__(module_name)
    except ModuleNotFoundError:
        missing.append(module_name)
print(" ".join(missing))
PY
)"

if [ -n "${missing_packages}" ]; then
  python3 -m pip install --no-input --default-timeout 60 ${missing_packages} \
    || python3 -m pip install --no-input ${missing_packages}
fi

python3 - <<'PY'
try:
    import torch  # noqa: F401
except ModuleNotFoundError as exc:
    raise SystemExit("phase2 运行需要 PyTorch，但当前环境缺少 torch。") from exc
PY

export PYTHONPATH="/workspace${PYTHONPATH:+:${PYTHONPATH}}"
export PHASE2_MAX_RUNTIME_SECONDS="${PHASE2_MAX_RUNTIME_SECONDS:-1740}"

python3 -m agent_framework.lora_optimize \
  --project-root /workspace \
  --time-budget-seconds "${PHASE2_MAX_RUNTIME_SECONDS}" | tee /workspace/results.log
