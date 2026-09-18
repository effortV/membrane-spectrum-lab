"""Background work without Streamlit calls, with cooperative cancellation.

Cancellation prevents subsequent work; it never promises to cancel provider billing.
The shared paid gate is acquired before creating a worker, not after a UI rerun.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import threading
import time
from typing import Any, Callable
import uuid

from .security import safe_error
from .task_monitor import public_progress, summarize_result
from .utils import utc_now


class OperationStopped(RuntimeError):
    def __init__(self, message: str, *, timed_out: bool = False):
        super().__init__(message)
        self.timed_out = timed_out


class TaskContext:
    def __init__(self, manager: "BackgroundTasks", task_id: str):
        self.manager = manager
        self.task_id = task_id

    def update(self, **progress: Any) -> None:
        self.manager.update(self.task_id, progress)

    def check(self) -> None:
        with self.manager._lock:
            job = self.manager._jobs[self.task_id]
            if job["cancel"].is_set():
                raise OperationStopped("已请求停止：已保存完成结果，不再发送后续请求。")
            if time.monotonic() - job["started_monotonic"] >= job["timeout"]:
                raise OperationStopped(
                    "已达到本批次时间上限：保留检查点，请分批继续。", timed_out=True
                )


class BackgroundTasks:
    def __init__(self, paid_lock: threading.Lock, paid_state: dict[str, str]):
        self.paid_lock = paid_lock
        self.paid_state = paid_state
        self._lock = threading.RLock()
        self._jobs: dict[str, dict[str, Any]] = {}

    def start(
        self,
        owner: str,
        label: str,
        kind: str,
        worker: Callable[[TaskContext], Any],
        *,
        timeout: float = 600.0,
        journal_root: Path | None = None,
        on_start: Callable[[str], None] | None = None,
    ) -> str:
        if not self.paid_lock.acquire(blocking=False):
            raise RuntimeError(
                f"另一个任务正在运行：{self.paid_state.get('label', '后台任务')}。未重复提交。"
            )
        task_id = uuid.uuid4().hex
        self.paid_state["label"] = label
        now = time.monotonic()
        job = {
            "task_id": task_id,
            "owner": owner,
            "label": label,
            "kind": kind,
            "status": "running",
            "created_at": utc_now(),
            "started_monotonic": now,
            "updated_monotonic": now,
            "updated_at": utc_now(),
            "events": [{"at": utc_now(), "stage": "准备任务"}],
            "timeout": timeout,
            "cancel": threading.Event(),
            "progress": {"stage": "准备任务"},
            "result": None,
            "error": None,
            "journal_root": journal_root,
        }
        with self._lock:
            self._jobs[task_id] = job
            finished = [key for key, item in self._jobs.items() if item["status"] != "running"]
            for key in finished[:-20]:
                del self._jobs[key]
        try:
            if on_start is not None:
                on_start(task_id)
            self._write_journal(job)
            thread = threading.Thread(
                target=self._run, args=(task_id, worker), daemon=True, name=f"xps-{task_id[:8]}"
            )
            thread.start()
        except BaseException as exc:
            with self._lock:
                job.update(
                    status="failed",
                    error=safe_error(exc),
                    finished_at=utc_now(),
                    finished_monotonic=time.monotonic(),
                )
                self._write_journal(job)
                self._jobs.pop(task_id, None)
            self.paid_state.clear()
            self.paid_lock.release()
            raise
        return task_id

    def _run(self, task_id: str, worker: Callable[[TaskContext], Any]) -> None:
        try:
            context = TaskContext(self, task_id)
            context.check()
            result = worker(context)
            status = "completed"
            if isinstance(result, dict) and result.get("interrupted"):
                status = "timed_out" if result.get("timed_out") else "cancelled"
            with self._lock:
                self._jobs[task_id].update(status=status, result=result)
        except OperationStopped as exc:
            with self._lock:
                self._jobs[task_id].update(
                    status="timed_out" if exc.timed_out else "cancelled", error=safe_error(exc)
                )
        except BaseException as exc:
            with self._lock:
                self._jobs[task_id].update(status="failed", error=safe_error(exc))
        finally:
            with self._lock:
                job = self._jobs[task_id]
                job["finished_at"] = utc_now()
                job["finished_monotonic"] = time.monotonic()
                job["events"].append({"at": job["finished_at"], "stage": job["status"]})
                self._write_journal(job)
            self.paid_state.clear()
            self.paid_lock.release()

    def update(self, task_id: str, progress: dict[str, Any]) -> None:
        with self._lock:
            job = self._jobs[task_id]
            changed = any(
                key in progress and progress[key] != job["progress"].get(key)
                for key in ("stage", "document_title", "page", "outer_fold")
            )
            job["progress"].update(deepcopy(progress))
            job["updated_monotonic"] = time.monotonic()
            job["updated_at"] = utc_now()
            if changed:
                job["events"].append({"at": job["updated_at"], **public_progress(job["progress"])})
                job["events"] = job["events"][-40:]
            self._write_journal(job)

    def cancel(self, task_id: str, owner: str) -> None:
        with self._lock:
            job = self._jobs.get(task_id)
            if not job or job["owner"] != owner:
                raise ValueError("只能停止当前会话提交的任务。")
            job["cancel"].set()

    def snapshot(self, owner: str) -> dict[str, Any] | None:
        with self._lock:
            active = [job for job in self._jobs.values() if job["status"] == "running"]
            own = [job for job in self._jobs.values() if job["owner"] == owner]
            if not active and not own:
                return None
            job = active[-1] if active else own[-1]
            is_owner = job["owner"] == owner
            now = job.get("finished_monotonic", time.monotonic())
            result = {
                "task_id": job["task_id"],
                "label": job["label"],
                "kind": job["kind"],
                "status": job["status"],
                "is_owner": is_owner,
                "cancel_requested": job["cancel"].is_set(),
                "created_at": job["created_at"],
                "elapsed_seconds": round(now - job["started_monotonic"], 1),
                "phase_elapsed_seconds": round(now - job["updated_monotonic"], 1),
                "progress": deepcopy(job["progress"])
                if is_owner
                else {"stage": "其他会话任务进行中"},
                "result": deepcopy(job["result"]) if is_owner else None,
                "error": job["error"] if is_owner else None,
            }
            return result

    @staticmethod
    def _write_journal(job: dict[str, Any]) -> None:
        root = job["journal_root"]
        if root is None:
            return
        # Diagnostic metadata only: no question, model prompt, response, or credentials.
        payload = {
            key: job.get(key)
            for key in (
                "task_id",
                "label",
                "kind",
                "status",
                "created_at",
                "finished_at",
                "error",
            )
        }
        payload["progress"] = public_progress(job["progress"])
        payload["events"] = job.get("events", [])[-40:]
        payload["updated_at"] = job.get("updated_at")
        payload["elapsed_seconds"] = round(
            job.get("finished_monotonic", time.monotonic()) - job["started_monotonic"], 1
        )
        payload["result_summary"] = summarize_result(job.get("result"))
        for key, value in list(payload.items()):
            if isinstance(value, str):
                payload[key] = safe_error(value)
        try:
            root.mkdir(parents=True, exist_ok=True)
            path = root / f"{job['task_id']}.json"
            temp = path.with_suffix(".tmp")
            temp.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temp.replace(path)
        except (OSError, ValueError, TypeError):
            # A log failure must not strand the global gate or kill useful work.
            pass
