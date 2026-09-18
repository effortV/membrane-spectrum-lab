"""Start API BEFORE the first UI browser visit, in the SAME Streamlit process."""

import argparse
import os
from pathlib import Path
import sys

from xps_agent.server_runtime import get_runtime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--address", default="127.0.0.1")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.environ["XPS_PROJECT_ROOT"] = str(root)
    runtime = get_runtime(root)
    runtime.start_api()
    from streamlit.web import cli

    sys.argv = [
        "streamlit",
        "run",
        str(root / "streamlit_app.py"),
        "--server.address",
        args.address,
        "--server.port",
        str(args.port),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]
    try:
        cli.main()
    finally:
        if runtime._api is not None:
            runtime._api.server.should_exit = True
            runtime._api.thread.join(timeout=3)


if __name__ == "__main__":
    main()
