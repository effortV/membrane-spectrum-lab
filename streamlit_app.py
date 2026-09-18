r"""Streamlit entry point: .venv\Scripts\python.exe -m streamlit run streamlit_app.py"""

import os
from pathlib import Path

os.environ.setdefault("XPS_PROJECT_ROOT", str(Path(__file__).resolve().parent))

from xps_agent.dashboard import main  # noqa: E402


main()
