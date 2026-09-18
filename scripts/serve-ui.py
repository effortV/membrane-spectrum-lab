"""Hidden service helper with plain UTF-8 logs, not PS5 formatted native stderr."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

from xps_agent.config import Settings

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=8501)
parser.add_argument("--address", default="127.0.0.1")
args = parser.parse_args()
project = Path(__file__).resolve().parents[1]
os.environ["XPS_PROJECT_ROOT"] = str(project)
settings = Settings.load(project)
logs = settings.workspace_root / "logs"
logs.mkdir(parents=True, exist_ok=True)
with (
    (logs / "streamlit-v03-active.out.log").open("ab") as output,
    (logs / "streamlit-v03-active.err.log").open("ab") as error,
):
    process = subprocess.run(
        [
            sys.executable,
            str(project / "scripts" / "run-server.py"),
            "--address",
            args.address,
            "--port",
            str(args.port),
        ],
        cwd=project,
        stdout=output,
        stderr=error,
        env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
    )
sys.exit(process.returncode)
