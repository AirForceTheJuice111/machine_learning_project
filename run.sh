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

export PYTHONPATH="/workspace${PYTHONPATH:+:${PYTHONPATH}}"

python3 -m agent_framework.evaluate \
  --target-spec /target/target_spec.json \
  --output /workspace/output.json \
  --skip-details-output | tee /workspace/results.log
