"""Durable research conversations; never depend on a browser session to save answers."""

from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any

from .db import StateDB
from .security import safe_error
from .utils import utc_now


RUNTIME_ID = uuid.uuid4().hex
MAX_HISTORY_TURNS = 20
MAX_HISTORY_CHARS = 80000
STATUS_LABELS = {
    "running": "正在回答",
    "completed": "已完成",
    "failed": "失败",
    "cancelled": "已停止",
    "timed_out": "超过时间上限",
    "interrupted": "服务重启中断",
}


def validate_text(text: str, *, maximum: int, label: str, allow_empty: bool = False) -> str:
    if not isinstance(text, str):
        raise ValueError(f"{label}必须是文本。")
    text = text.strip()
    if (not text and not allow_empty) or len(text) > maximum:
        raise ValueError(
            f"{label}最多 {maximum} 个字符" + ("。" if allow_empty else "且不能为空。")
        )
    if safe_error(text, limit=maximum) != text:
        raise ValueError(f"{label}可能包含密钥，请移除凭据后重试。")
    return text


class ConversationStore:
    def __init__(self, db: StateDB, *, runtime_id: str = RUNTIME_ID):
        self.db = db
        self.runtime_id = runtime_id

    def create(self, title: str = "新研究会话") -> str:
        title = validate_text(title, maximum=120, label="会话名称")
        key, now = uuid.uuid4().hex, utc_now()
        with self.db.connect() as con:
            con.execute(
                "INSERT INTO chat_conversations(conversation_id,title,created_at,updated_at) VALUES (?,?,?,?)",
                (key, title, now, now),
            )
        return key

    def get(self, key: str) -> dict[str, Any]:
        rows = self.db.rows("SELECT * FROM chat_conversations WHERE conversation_id=?", (key,))
        if not rows:
            raise ValueError("会话不存在，请选择已保存的会话。")
        return rows[0]

    def list(self, *, archived: bool = False) -> list[dict[str, Any]]:
        return self.db.rows(
            """SELECT c.*, COUNT(t.turn_id) AS turns,
                      SUM(CASE WHEN t.status='running' THEN 1 ELSE 0 END) AS running
               FROM chat_conversations c LEFT JOIN chat_turns t USING(conversation_id)
               WHERE c.archived=? GROUP BY c.conversation_id ORDER BY c.updated_at DESC,c.rowid DESC""",
            (int(archived),),
        )

    def turns(self, key: str) -> list[dict[str, Any]]:
        return self.db.rows(
            "SELECT * FROM chat_turns WHERE conversation_id=? ORDER BY rowid", (key,)
        )

    def update(self, key: str, *, title: str, memory: str) -> None:
        title = validate_text(title, maximum=120, label="会话名称")
        memory = validate_text(memory, maximum=6000, label="长期研究备注", allow_empty=True)
        self.get(key)
        with self.db.connect() as con:
            con.execute(
                "UPDATE chat_conversations SET title=?,memory=?,updated_at=? WHERE conversation_id=?",
                (title, memory, utc_now(), key),
            )

    def archive(self, key: str, archived: bool = True) -> None:
        with self.db.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            if con.execute(
                "SELECT 1 FROM chat_turns WHERE conversation_id=? AND status='running'", (key,)
            ).fetchone():
                raise ValueError("会话正在回答，完成或停止后才能归档。")
            con.execute(
                "UPDATE chat_conversations SET archived=?,updated_at=? WHERE conversation_id=?",
                (int(archived), utc_now(), key),
            )

    def context(self, key: str) -> dict[str, Any]:
        conversation = self.get(key)
        completed = [turn for turn in self.turns(key) if turn["status"] == "completed"]
        selected: list[dict[str, Any]] = []
        chars = 0
        for turn in reversed(completed):
            size = len(turn["question"]) + len(turn["answer"])
            if len(selected) >= MAX_HISTORY_TURNS or chars + size > MAX_HISTORY_CHARS:
                break
            selected.append(turn)
            chars += size
        selected.reverse()
        return {
            "history": [
                message
                for turn in selected
                for message in (
                    {"role": "user", "content": turn["question"]},
                    {"role": "assistant", "content": turn["answer"]},
                )
            ],
            "memory": conversation["memory"],
            "history_turns": len(selected),
            "omitted_turns": len(completed) - len(selected),
        }

    def reserve(self, key: str, question: str, *, model: str, task_id: str) -> str:
        question = validate_text(question, maximum=6000, label="问题")
        turn_id, now = uuid.uuid4().hex, utc_now()
        try:
            with self.db.connect() as con:
                con.execute("BEGIN IMMEDIATE")
                conversation = con.execute(
                    "SELECT * FROM chat_conversations WHERE conversation_id=?", (key,)
                ).fetchone()
                if conversation is None or conversation["archived"]:
                    raise ValueError("请选择未归档的会话后提问。")
                con.execute(
                    """INSERT INTO chat_turns
                       (turn_id,conversation_id,question,status,model,task_id,runtime_id,created_at)
                       VALUES (?,?,?,'running',?,?,?,?)""",
                    (turn_id, key, question, model, task_id, self.runtime_id, now),
                )
                title = (
                    question[:60]
                    if conversation["title"] == "新研究会话"
                    else conversation["title"]
                )
                con.execute(
                    "UPDATE chat_conversations SET title=?,updated_at=? WHERE conversation_id=?",
                    (title, now, key),
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError("本会话已有问题正在回答，未重复提交。") from exc
        return turn_id

    def save_context(self, turn_id: str, context: dict[str, Any]) -> None:
        with self.db.connect() as con:
            con.execute(
                "UPDATE chat_turns SET history_turns=?,omitted_turns=?,memory_snapshot=? WHERE turn_id=? AND status='running'",
                (context["history_turns"], context["omitted_turns"], context["memory"], turn_id),
            )

    def finish(self, turn_id: str, answer: str) -> None:
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("模型没有返回完整回答，问题已保存；请查看失败记录。")
        # Store the complete public answer, never private reasoning or credentials.
        answer = safe_error(answer, limit=len(answer))
        with self.db.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            changed = con.execute(
                "UPDATE chat_turns SET status='completed',answer=?,error=NULL,finished_at=? WHERE turn_id=? AND status='running' AND runtime_id=?",
                (answer, utc_now(), turn_id, self.runtime_id),
            ).rowcount
            if changed != 1:
                raise RuntimeError("会话轮次状态已改变，未覆盖已有回答。")
            con.execute(
                "UPDATE chat_conversations SET updated_at=? WHERE conversation_id=(SELECT conversation_id FROM chat_turns WHERE turn_id=?)",
                (utc_now(), turn_id),
            )

    def fail(self, turn_id: str, error: object, *, status: str = "failed") -> None:
        if status not in {"failed", "cancelled", "timed_out", "interrupted"}:
            raise ValueError("Invalid conversation failure status")
        with self.db.connect() as con:
            con.execute(
                "UPDATE chat_turns SET status=?,error=?,finished_at=? WHERE turn_id=? AND status='running' AND runtime_id=?",
                (status, safe_error(error), utc_now(), turn_id, self.runtime_id),
            )

    def recover(self, journal_root: Path | None = None) -> None:
        """Mark orphaned requests, without retrying or inferring provider billing."""
        with self.db.connect() as con:
            con.execute(
                "UPDATE chat_turns SET status='interrupted',error=?,finished_at=? WHERE status='running' AND runtime_id<>?",
                (
                    "服务已重启，上一请求的完成情况未知。未自动重发；原问题保留。",
                    utc_now(),
                    self.runtime_id,
                ),
            )
        if journal_root is None:
            return
        for turn in self.db.rows("SELECT turn_id,task_id FROM chat_turns WHERE status='running'"):
            try:
                task = json.loads(
                    (journal_root / f"{turn['task_id']}.json").read_text(encoding="utf-8")
                )
            except (OSError, ValueError):
                continue
            status = task.get("status")
            if status in {"failed", "cancelled", "timed_out", "completed"}:
                self.fail(
                    turn["turn_id"],
                    task.get("error") or "后台任务已结束，但未保存完整回答。未自动重发。",
                    status="failed" if status == "completed" else status,
                )

    def export(self, key: str) -> dict[str, Any]:
        return {"format_version": 1, "conversation": self.get(key), "turns": self.turns(key)}

    def markdown(self, key: str) -> str:
        from .presentation import normalize_math_markdown

        conversation = self.get(key)
        parts = [f"# {conversation['title']}", f"会话 ID：{key}"]
        if conversation["memory"]:
            parts += ["## 长期研究备注", conversation["memory"]]
        for turn in self.turns(key):
            parts += [f"## 用户 · {turn['created_at']}", turn["question"], "## 智能体"]
            parts.append(
                normalize_math_markdown(turn["answer"])
                or f"{STATUS_LABELS[turn['status']]}：{turn['error'] or '等待模型完整返回。'}"
            )
        return "\n\n".join(parts) + "\n"


def start_conversation_task(
    settings, db: StateDB, manager, owner: str, key: str, question: str
) -> str:
    from .agent import ResearchAgent
    from .llm import SiliconFlowClient
    from .tasks import OperationStopped

    question = validate_text(question, maximum=6000, label="问题")
    store = ConversationStore(db)
    reservation: dict[str, str] = {}

    def reserve(task_id: str) -> None:
        reservation["turn_id"] = store.reserve(
            key, question, model=settings.main_model, task_id=task_id
        )

    def worker(context):
        turn_id = reservation["turn_id"]
        agent = None
        try:
            previous = store.context(key)
            store.save_context(turn_id, previous)
            context.update(
                stage="读取会话上下文",
                history_turns=previous["history_turns"],
                omitted_turns=previous["omitted_turns"],
                conversation_id=key,
            )
            with SiliconFlowClient(
                settings, db, progress=context.update, check_cancelled=context.check
            ) as llm:
                agent = ResearchAgent(settings, db, llm)
                answer = agent.ask(
                    question,
                    history=previous["history"],
                    memory=previous["memory"],
                    omitted_turns=previous["omitted_turns"],
                )
                store.finish(turn_id, answer)
            return {"conversation_id": key, "turn_id": turn_id}
        except BaseException as exc:
            status = "failed"
            if isinstance(exc, OperationStopped):
                status = "timed_out" if exc.timed_out else "cancelled"
            store.fail(turn_id, exc, status=status)
            raise
        finally:
            if agent is not None:
                agent.literature.client.close()

    try:
        return manager.start(
            owner,
            "智能体问答",
            "ask",
            worker,
            on_start=reserve,
            timeout=settings.action_timeout_seconds,
            journal_root=settings.workspace_root / "state" / "tasks",
        )
    except BaseException as exc:
        if reservation.get("turn_id"):
            store.fail(reservation["turn_id"], exc)
        raise
