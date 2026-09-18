"""Generate backend credential on the server without ever displaying its value."""

import argparse
from pathlib import Path
import secrets

from xps_agent.config import Settings
from xps_agent.security import secure_local_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    settings = Settings.load(args.project_root)
    directory = settings.workspace_root / "state" / "connections"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "backend_token.txt"
    if path.exists():
        if len(path.read_text(encoding="utf-8").strip()) < 32:
            raise RuntimeError("Existing token is invalid; do not overwrite automatically.")
        secure_local_file(path)
        print("Backend credential already exists; preserved, value hidden.")
        return
    # Secure the empty file BEFORE adding credential contents.
    with path.open("x", encoding="utf-8"):
        pass
    secure_local_file(path)
    path.write_text(secrets.token_urlsafe(48) + "\n", encoding="utf-8")
    print("Backend credential created on server; value hidden.")


if __name__ == "__main__":
    main()
