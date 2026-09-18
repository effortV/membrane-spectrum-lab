"""Explicit migration; never runs during a Streamlit rerun."""

import argparse
from pathlib import Path
from xps_agent.storage import migrate_storage
from xps_agent.utils import json_dumps

parser = argparse.ArgumentParser()
parser.add_argument("--apply", action="store_true")
parser.add_argument("--server-stopped", action="store_true")
arguments = parser.parse_args()
print(
    json_dumps(
        migrate_storage(
            Path(__file__).resolve().parents[1],
            Path(r"D:\data\XPS-914"),
            Path(r"D:\data\XPS-agent"),
            apply=arguments.apply,
            server_stopped=arguments.server_stopped,
        )
    )
)
