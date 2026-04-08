import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = BASE_DIR / "workspace"
TEMPLATES_DIR = BASE_DIR / "templates"
RESULTS_FILE = BASE_DIR / "results.json"
REASONING_LOG_FILE = BASE_DIR / "agent_reasoning.log"

MAX_RETRIES = 3
