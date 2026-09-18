from __future__ import annotations

import json
from typing import Any, Callable

from .config import Settings
from .db import StateDB
from .literature import LiteratureService
from .llm import SiliconFlowClient
from .utils import json_dumps
from .security import safe_error


TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_experiment",
            "description": "Read one saved ML relationship experiment by its exact ID, including cohort, variables and grouped validation results.",
            "parameters": {
                "type": "object",
                "properties": {"experiment_id": {"type": "string", "maxLength": 64}},
                "required": ["experiment_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_column_inventory",
            "description": "Read measured variable names, roles, coverage, and unit warnings. No sample outcomes are exposed for feature selection.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_health_report",
            "description": "Read offline runtime, stale-artifact, citation, DOI, and linkage diagnostics.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_data_profile",
            "description": "Read the current NF/RO workbook and spectrum audit profile.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_evidence",
            "description": "Search page-located evidence extracted from local literature.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 50},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_hypotheses",
            "description": "List candidate descriptor hypotheses and their statuses.",
            "parameters": {
                "type": "object",
                "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 50}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_experiments",
            "description": "List ML validation runs and evidence-review decisions.",
            "parameters": {
                "type": "object",
                "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 50}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "discover_literature",
            "description": "Search OpenAlex and add results to the download queue.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer", "minimum": 1, "maximum": 100},
                },
                "required": ["query"],
            },
        },
    },
]


class ResearchAgent:
    def __init__(self, settings: Settings, db: StateDB, llm: SiliconFlowClient):
        self.settings = settings
        self.db = db
        self.llm = llm
        self.literature = LiteratureService(settings, db)
        self.system_prompt = (settings.project_root / "prompts" / "research_agent.md").read_text(
            encoding="utf-8"
        )
        self.handlers: dict[str, Callable[..., Any]] = {
            "get_column_inventory": self.get_column_inventory,
            "get_health_report": self.get_health_report,
            "get_data_profile": self.get_data_profile,
            "search_evidence": self.search_evidence,
            "list_hypotheses": self.list_hypotheses,
            "list_experiments": self.list_experiments,
            "get_experiment": self.get_experiment,
            "discover_literature": self.discover_literature,
        }

    @staticmethod
    def _limit(value: int, maximum: int = 50) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("Tool limit must be an integer")
        return max(1, min(value, maximum))

    def get_column_inventory(self) -> list[dict[str, Any]]:
        import pandas as pd
        from .research import current_research_table, field_catalog, load_role_overrides

        path = current_research_table(self.settings)
        if not path.exists():
            return []
        symbols = [
            json.loads(row["spec_json"]).get("symbol")
            for row in self.db.rows("SELECT spec_json FROM hypotheses WHERE status='calculable'")
        ]
        return field_catalog(
            pd.read_csv(path, low_memory=False), load_role_overrides(self.settings), symbols
        ).to_dict("records")

    def get_health_report(self) -> dict[str, Any]:
        from .diagnostics import health_report

        return health_report(self.settings, self.db)

    def get_data_profile(self) -> dict[str, Any]:
        path = self.settings.workspace_root / "canonical" / "data_profile.json"
        if not path.exists():
            return {"error": "No data profile. Run audit-data first."}
        return json.loads(path.read_text(encoding="utf-8"))

    def search_evidence(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        if not isinstance(query, str) or len(query) > 512:
            raise ValueError("Evidence query must be text of at most 512 characters")
        pattern = f"%{query}%"
        rows = self.db.rows(
            """
            SELECT e.evidence_id, e.doc_key, e.page, e.kind, e.claim, e.locator,
                   e.confidence, e.payload_json, d.title, d.doi
            FROM evidence e JOIN documents d USING(doc_key)
            WHERE e.claim LIKE ? OR d.title LIKE ?
            ORDER BY e.confidence DESC LIMIT ?
            """,
            (pattern, pattern, self._limit(limit)),
        )
        for row in rows:
            payload = json.loads(row.pop("payload_json"))
            row["conditions"] = payload.get("conditions", [])
            row["ambiguities"] = payload.get("ambiguities", [])
            row["directly_visible"] = payload.get("directly_visible")
            row["verification"] = payload.get("verification", "legacy_model_extracted_unreviewed")
        return rows

    def list_hypotheses(self, limit: int = 20) -> list[dict[str, Any]]:
        return self.db.rows(
            """
            SELECT hypothesis_id, name, status, spec_json, model, created_at
            FROM hypotheses ORDER BY created_at DESC LIMIT ?
            """,
            (self._limit(limit),),
        )

    def list_experiments(self, limit: int = 20) -> list[dict[str, Any]]:
        return self.db.rows(
            """
            SELECT experiment_id, hypothesis_id, run_dir, data_hash, split_hash,
                   decision, created_at, metrics_json FROM experiments ORDER BY created_at DESC LIMIT ?
            """,
            (self._limit(limit),),
        )

    def get_experiment(self, experiment_id: str) -> dict[str, Any]:
        if not isinstance(experiment_id, str) or not 1 <= len(experiment_id) <= 64:
            raise ValueError("Invalid experiment ID")
        rows = self.db.rows(
            "SELECT experiment_id,run_dir,decision,metrics_json,created_at FROM experiments WHERE experiment_id=?",
            (experiment_id,),
        )
        if not rows:
            return {"error": "Experiment not found"}
        result = rows[0]
        result["research_result"] = json.loads(result.pop("metrics_json"))
        return result

    def discover_literature(self, query: str, max_results: int = 20) -> dict[str, Any]:
        keys = self.literature.search_openalex(query, self._limit(max_results, 100))
        works = []
        for key in keys[:20]:
            works.extend(
                self.db.rows(
                    "SELECT doc_key,title,doi,year,landing_url,status FROM documents WHERE doc_key=?",
                    (key,),
                )
            )
        return {
            "query": query,
            "added_to_queue": len(keys),
            "works": works,
            "novelty_policy": "Search hits/absence are not proof of novelty; inspect full text and synonyms.",
        }

    def ask(
        self,
        question: str,
        max_steps: int = 8,
        *,
        history: list[dict[str, str]] | None = None,
        memory: str = "",
        omitted_turns: int = 0,
    ) -> str:
        if not question.strip() or len(question) > 6000:
            raise ValueError("问题不能为空且最多 6000 个字符。")
        if safe_error(question, limit=6000) != question:
            raise ValueError("问题可能包含凭据或过长敏感内容，请移除密钥后重试。")
        messages: list[dict[str, Any]] = [{"role": "system", "content": self.system_prompt}]
        if history or memory:
            messages[0]["content"] += (
                "\n同一研究会话的前文用于理解指代与研究约束，但旧回答不是独立证据。"
                "涉及定量结果、文献或新颖性时，按证据 ID / 实验 ID 重新查询工具核验；"
                "不要将前文推测升级为事实，也不要假装记得未提供的历史。"
            )
        if memory:
            if len(memory) > 6000 or safe_error(memory, limit=6000) != memory:
                raise ValueError("长期研究备注最多 6000 字符且不能包含密钥。")
            messages.append({"role": "user", "content": "本会话长期研究备注：\n" + memory})
        history_chars = 0
        for previous in history or []:
            role, content = previous.get("role"), previous.get("content")
            if role not in {"user", "assistant"} or not isinstance(content, str):
                raise ValueError("历史只能包含已保存的用户问题和完整助手回答。")
            history_chars += len(content)
            if history_chars > 80000 or safe_error(content, limit=len(content)) != content:
                raise ValueError("历史超出上下文窗口或包含凭据，请使用长期研究备注。")
            messages.append({"role": role, "content": content})
        if omitted_turns:
            messages.append(
                {
                    "role": "user",
                    "content": f"上下文窗口说明：更早的 {omitted_turns} 轮未带入本次请求，仍完整保存在会话历史中。若需要其中细节，请要求我明确补充。",
                }
            )
        messages.append({"role": "user", "content": question})
        steps = max(1, min(max_steps, 8))
        for step in range(steps):
            final_step = steps > 1 and step == steps - 1
            if final_step:
                messages.append(
                    {
                        "role": "user",
                        "content": "本轮工具预算已到上限。请仅依据已经取得的证据直接回答原问题；明确未知信息和后续验证，不再请求工具。",
                    }
                )
            body = self.llm.chat(
                messages,
                model=self.settings.main_model,
                max_tokens=5000,
                temperature=0.1,
                tools=None if final_step else TOOLS,
                use_cache=False,
            )
            message = body["choices"][0]["message"]
            messages.append(message)
            tool_calls = message.get("tool_calls") or []
            if not isinstance(tool_calls, list) or len(tool_calls) > 8:
                return "模型单轮工具调用超出安全上限，未执行本轮工具；请缩小问题。"
            if final_step and tool_calls:
                return "已达到本轮工具预算，模型未给出最终回答。请缩小问题；已有会话历史仍保留。"
            if not tool_calls:
                return str(message.get("content") or "")
            for tool_call in tool_calls:
                function = tool_call.get("function") or {}
                name = function.get("name")
                try:
                    arguments = json.loads(function.get("arguments") or "{}")
                    if not isinstance(arguments, dict):
                        raise ValueError("Tool arguments must be a JSON object")
                    handler = self.handlers.get(name)
                    if handler is None:
                        result: Any = {"error": f"Unknown tool: {name}"}
                    else:
                        result = handler(**arguments)
                except Exception as exc:
                    result = {"error": safe_error(exc)}
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.get("id"),
                        "name": name,
                        "content": json_dumps(result),
                    }
                )
        return "达到工具调用步数上限；请缩小问题或先补充证据。"
