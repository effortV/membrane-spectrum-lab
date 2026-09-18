"""One process-wide runtime shared by the local UI and embedded FastAPI."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import threading

from .config import Settings
from .db import StateDB
from .tasks import BackgroundTasks


class ServerRuntime:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.settings = Settings.load(project_root)
        self.settings.ensure_workspace()
        self.db = StateDB(self.settings.database_path)
        self.db.initialize()
        self.paid_lock = threading.Lock()
        self.paid_state: dict[str, str] = {}
        self.manager = BackgroundTasks(self.paid_lock, self.paid_state)
        self._api_lock = threading.RLock()
        self._api = None

    def context(self):
        latest = Settings.load(self.project_root)
        if latest.database_path.resolve() != self.db.path.resolve():
            raise RuntimeError("数据目录已变化，请在任务结束后重启服务器服务。")
        return latest, self.db

    def start_api(self):
        from .backend import start_embedded_api

        with self._api_lock:
            if self._api is None:
                settings, db = self.context()
                self._api = start_embedded_api(settings, db, self.manager)
            elif not self._api.thread.is_alive():
                raise RuntimeError("服务器接口已停止，请核查后重启服务；未自动重新启动。")
            return self._api


@lru_cache(maxsize=16)
def _runtime(project_root: str):
    return ServerRuntime(Path(project_root))


def get_runtime(project_root: Path | None = None):
    root = Settings.load(project_root).project_root
    return _runtime(str(root.resolve()))
