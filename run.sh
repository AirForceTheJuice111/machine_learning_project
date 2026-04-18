#!/usr/bin/env bash
set -euo pipefail

cd /workspace

python3 - <<'PY'
import importlib
import subprocess
import sys

missing = []
for module_name, package_name in [("openai", "openai"), ("rich", "rich")]:
    try:
        importlib.import_module(module_name)
    except Exception:
        missing.append(package_name)

if missing:
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            *missing,
            "-i",
            "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple",
            "--default-timeout",
            "0.3",
        ]
    )
PY

python3 -m agent_framework.evaluate \
  --target-spec /target/target_spec.json \
  --output /workspace/output.json \
  --skip-details-output
