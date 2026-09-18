from __future__ import annotations

import os
import math
import re
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from dotenv import load_dotenv

from .security import hash_ui_password, secure_local_file


API_SETTING_NAMES = frozenset(
    {
        "SILICONFLOW_API_KEY",
        "OPENALEX_API_KEY",
        "OPENALEX_MAILTO",
        "ELSEVIER_API_KEY",
        "ELSEVIER_INSTTOKEN",
        "XPS_UI_PASSWORD_HASH",
    }
)
RUNTIME_SETTING_NAMES = frozenset(
    {
        "XPS_STORAGE_ROOT",
        "XPS_WORKSPACE_ROOT",
        "XPS_DATA_ROOT",
        "XPS_REFERENCE_ROOT",
        "XPS_LEGACY_ROOT",
    }
)
_CONFIG_LOCK = threading.Lock()
MODEL_SETTING_RANGES = {
    "XPS_LLM_TIMEOUT_SECONDS": (30, 1800),
    "XPS_VISION_TIMEOUT_SECONDS": (30, 1800),
    "XPS_REQUEST_TIMEOUT_SECONDS": (120, 7200),
    "XPS_ACTION_TIMEOUT_SECONDS": (300, 14400),
    "XPS_LLM_TIMEOUT_RETRIES": (0, 1),
    "XPS_VISION_TIMEOUT_RETRIES": (0, 1),
}
MODEL_SETTING_NAMES = frozenset({*MODEL_SETTING_RANGES, "XPS_LLM_STREAM_ENABLED"})


def save_api_settings(project_root: Path, updates: Mapping[str, str]) -> list[str]:
    """Atomically update the allow-listed API settings without exposing values."""
    with _CONFIG_LOCK:
        return _save_api_settings(project_root, updates)


def save_runtime_settings(project_root: Path, updates: Mapping[str, str]) -> list[str]:
    if set(updates) - RUNTIME_SETTING_NAMES:
        raise ValueError("Unsupported storage configuration")
    with _CONFIG_LOCK:
        return _save_api_settings(project_root, updates, allowed=RUNTIME_SETTING_NAMES)


def save_model_settings(project_root: Path, updates: Mapping[str, str]) -> list[str]:
    if set(updates) - MODEL_SETTING_NAMES:
        raise ValueError("Unsupported model configuration")
    for name, value in updates.items():
        if name == "XPS_LLM_STREAM_ENABLED":
            if value not in {"true", "false"}:
                raise ValueError("Streaming must be true or false")
            continue
        try:
            number = float(value)
        except ValueError:
            raise ValueError(f"Invalid numeric model setting: {name}") from None
        low, high = MODEL_SETTING_RANGES[name]
        if not math.isfinite(number) or not low <= number <= high:
            raise ValueError(f"Model setting outside range: {name}")
        if name.endswith("RETRIES") and number != int(number):
            raise ValueError("Retry count must be an integer")
    with _CONFIG_LOCK:
        return _save_api_settings(project_root, updates, allowed=MODEL_SETTING_NAMES)


def save_ui_password(project_root: Path, password: str) -> str:
    hashed = hash_ui_password(password)
    save_api_settings(project_root, {"XPS_UI_PASSWORD_HASH": hashed})
    return hashed


def _save_api_settings(
    project_root: Path, updates: Mapping[str, str], *, allowed=API_SETTING_NAMES
) -> list[str]:
    unsupported = set(updates) - allowed
    if unsupported:
        raise ValueError(f"Unsupported API setting names: {', '.join(sorted(unsupported))}")

    cleaned: dict[str, str] = {}
    for name, value in updates.items():
        if "\r" in value or "\n" in value or "\x00" in value:
            raise ValueError(f"{name} contains an invalid control character")
        cleaned[name] = value.strip()
    if not cleaned:
        return []

    env_path = project_root.resolve() / ".env"
    example_path = project_root.resolve() / ".env.example"
    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8-sig").splitlines()
    elif example_path.exists():
        lines = example_path.read_text(encoding="utf-8-sig").splitlines()
    else:
        lines = []

    pending = dict(cleaned)
    output: list[str] = []
    for line in lines:
        match = re.match(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=", line)
        name = match.group(1) if match else None
        if name in cleaned:
            if name not in pending:
                continue  # Remove duplicate assignments instead of allowing old values to win.
            value = pending.pop(name)
            escaped = value.replace("\\", "\\\\").replace('"', '\\"')
            output.append(f'{name}="{escaped}"' if value else f"{name}=")
        else:
            output.append(line)
    for name, value in pending.items():
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        output.append(f'{name}="{escaped}"' if value else f"{name}=")

    with tempfile.NamedTemporaryFile(dir=env_path.parent, prefix=".env.", delete=False) as handle:
        temp_path = Path(handle.name)
    try:
        temp_path.write_text("\n".join(output) + "\n", encoding="utf-8", newline="\n")
        secure_local_file(temp_path)
        os.replace(temp_path, env_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()

    for name, value in cleaned.items():
        if value:
            os.environ[name] = value
        else:
            os.environ.pop(name, None)
    return list(cleaned)


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _bounded_float(name: str, default: float, low: float, high: float) -> float:
    value = _float(name, default)
    return max(low, min(high, value)) if math.isfinite(value) else default


@dataclass(frozen=True)
class Settings:
    project_root: Path
    data_root: Path
    reference_root: Path
    legacy_root: Path
    workspace_root: Path
    database_path: Path
    siliconflow_api_key: str | None
    siliconflow_base_url: str
    main_model: str
    vision_model: str
    openalex_api_key: str | None
    openalex_mailto: str | None
    elsevier_api_key: str | None
    elsevier_insttoken: str | None
    llm_timeout_seconds: float
    llm_max_retries: int
    vision_max_pages_per_pdf: int
    download_rate_seconds: float
    llm_max_requests_per_action: int = 24
    ui_password_hash: str | None = None
    vision_timeout_seconds: float = 600.0
    vision_max_attempts: int = 2
    action_timeout_seconds: float = 3600.0
    storage_root: Path | None = None
    llm_stream_enabled: bool = True
    request_timeout_seconds: float = 1800.0
    llm_timeout_retries: int = 0
    vision_timeout_retries: int = 0

    @classmethod
    def load(cls, project_root: Path | None = None) -> "Settings":
        root = (
            Path(project_root or os.getenv("XPS_PROJECT_ROOT") or Path.cwd()).expanduser().resolve()
        )
        load_dotenv(root / ".env", override=False)
        if project_root is None:
            root = Path(os.getenv("XPS_PROJECT_ROOT", str(root))).expanduser().resolve()
        storage = (
            Path(os.environ["XPS_STORAGE_ROOT"]).resolve()
            if os.getenv("XPS_STORAGE_ROOT")
            else None
        )
        workspace = Path(
            os.getenv("XPS_WORKSPACE_ROOT")
            or (storage / "workspace" if storage else root / "workspace")
        ).resolve()
        return cls(
            project_root=root,
            data_root=Path(os.getenv("XPS_DATA_ROOT", r"D:\data\XPS-914")).resolve(),
            reference_root=Path(
                os.getenv("XPS_REFERENCE_ROOT", r"D:\data\XPS-914\reference")
            ).resolve(),
            legacy_root=Path(os.getenv("XPS_LEGACY_ROOT", r"D:\zzh\XPS-agent\final")).resolve(),
            workspace_root=workspace,
            database_path=workspace / "state" / "xps_agent.sqlite3",
            siliconflow_api_key=os.getenv("SILICONFLOW_API_KEY") or None,
            siliconflow_base_url=os.getenv(
                "SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"
            ).rstrip("/"),
            main_model=os.getenv("XPS_MAIN_MODEL", "deepseek-ai/DeepSeek-V4-Pro"),
            vision_model=os.getenv("XPS_VISION_MODEL", "Qwen/Qwen3-VL-32B-Instruct"),
            openalex_api_key=os.getenv("OPENALEX_API_KEY") or None,
            openalex_mailto=os.getenv("OPENALEX_MAILTO") or None,
            elsevier_api_key=os.getenv("ELSEVIER_API_KEY") or None,
            elsevier_insttoken=os.getenv("ELSEVIER_INSTTOKEN") or None,
            llm_timeout_seconds=_bounded_float("XPS_LLM_TIMEOUT_SECONDS", 600.0, 30, 1800),
            llm_max_retries=_int("XPS_LLM_MAX_RETRIES", 4),
            vision_max_pages_per_pdf=_int("XPS_VISION_MAX_PAGES_PER_PDF", 4),
            download_rate_seconds=_float("XPS_DOWNLOAD_RATE_SECONDS", 1.0),
            llm_max_requests_per_action=max(1, _int("XPS_LLM_MAX_REQUESTS_PER_ACTION", 24)),
            ui_password_hash=os.getenv("XPS_UI_PASSWORD_HASH") or None,
            vision_timeout_seconds=_bounded_float("XPS_VISION_TIMEOUT_SECONDS", 600.0, 30, 1800),
            vision_max_attempts=max(1, min(3, _int("XPS_VISION_MAX_ATTEMPTS", 2))),
            action_timeout_seconds=_bounded_float("XPS_ACTION_TIMEOUT_SECONDS", 3600.0, 300, 14400),
            storage_root=storage,
            llm_stream_enabled=os.getenv("XPS_LLM_STREAM_ENABLED", "true").lower() == "true",
            request_timeout_seconds=_bounded_float("XPS_REQUEST_TIMEOUT_SECONDS", 1800, 120, 7200),
            llm_timeout_retries=max(0, min(1, _int("XPS_LLM_TIMEOUT_RETRIES", 0))),
            vision_timeout_retries=max(0, min(1, _int("XPS_VISION_TIMEOUT_RETRIES", 0))),
        )

    def ensure_workspace(self) -> None:
        for relative in (
            "state",
            "library",
            "inbox",
            "cache/pdf_text",
            "cache/pages",
            "cache/llm",
            "canonical",
            "runs",
            "logs",
            "reports",
        ):
            (self.workspace_root / relative).mkdir(parents=True, exist_ok=True)
        if self.storage_root:
            for relative in (
                "raw/tables/incoming",
                "raw/literature/NF",
                "raw/literature/RO",
                "raw/xps_uploads",
                "catalog",
            ):
                (self.storage_root / relative).mkdir(parents=True, exist_ok=True)

    @property
    def upload_root(self) -> Path:
        return (
            (self.storage_root / "raw" / "xps_uploads")
            if self.storage_root
            else (self.workspace_root / "inbox" / "xps")
        )

    @property
    def catalog_root(self) -> Path:
        return (
            self.storage_root / "catalog" if self.storage_root else self.workspace_root / "catalog"
        )

    def public_summary(self) -> dict[str, object]:
        return {
            "project_root": str(self.project_root),
            "storage_root": str(self.storage_root)
            if self.storage_root
            else str(self.workspace_root.parent),
            "data_root": str(self.data_root),
            "reference_root": str(self.reference_root),
            "legacy_root": str(self.legacy_root),
            "database_path": str(self.database_path),
            "main_model": self.main_model,
            "vision_model": self.vision_model,
            "siliconflow_key": "configured" if self.siliconflow_api_key else "missing",
            "openalex_key": "configured" if self.openalex_api_key else "missing/optional",
            "elsevier_key": "configured" if self.elsevier_api_key else "missing",
            "llm_max_requests_per_action": self.llm_max_requests_per_action,
            "llm_timeout_seconds": self.llm_timeout_seconds,
            "llm_stream_enabled": self.llm_stream_enabled,
            "request_timeout_seconds": self.request_timeout_seconds,
            "llm_timeout_retries": self.llm_timeout_retries,
            "vision_timeout_retries": self.vision_timeout_retries,
            "vision_timeout_seconds": self.vision_timeout_seconds,
            "vision_max_attempts": self.vision_max_attempts,
            "action_timeout_seconds": self.action_timeout_seconds,
        }
