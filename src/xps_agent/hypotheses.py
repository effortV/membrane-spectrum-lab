from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import Settings
from .db import StateDB
from .features import (
    DescriptorEngine,
    DescriptorSpec,
    SPECTRUM_SUMMARY_VERSION,
    is_known_descriptor,
    is_quality_column,
)
from .llm import SiliconFlowClient
from .utils import json_dumps, sha256_text


def validate_candidate(
    spec: DescriptorSpec,
    roles: dict[str, str],
    available_evidence: set[str],
    reserved_symbols: set[str],
) -> None:
    missing = [name for name in spec.inputs if name not in roles]
    forbidden = [
        name
        for name in spec.inputs
        if roles.get(name)
        in {
            "outcome",
            "performance",
            "structure_proxy",
            "test_condition",
            "identity",
            "quality",
            "spectrum_quality",
        }
        or is_known_descriptor(name)
        or is_quality_column(name)
    ]
    if missing:
        raise ValueError(f"Inputs absent from audited data: {missing}")
    if forbidden:
        raise ValueError(
            f"Outcome/identity/known/quality-artifact inputs are forbidden: {forbidden}"
        )
    if spec.symbol in reserved_symbols:
        raise ValueError(
            f"Descriptor symbol conflicts with an existing column/candidate: {spec.symbol}"
        )
    unknown_evidence = set(spec.evidence_ids) - available_evidence
    if unknown_evidence:
        raise ValueError(f"Evidence IDs do not exist: {sorted(unknown_evidence)}")
    if not spec.novelty_queries:
        raise ValueError("At least one novelty search query is required")


class HypothesisEngine:
    def __init__(self, settings: Settings, db: StateDB, llm: SiliconFlowClient | None = None):
        self.settings = settings
        self.db = db
        self.llm = llm
        self.system_prompt = (settings.project_root / "prompts" / "hypothesis_agent.md").read_text(
            encoding="utf-8"
        )

    def _context(self) -> dict[str, Any]:
        profile_path = self.settings.workspace_root / "canonical" / "data_profile.json"
        columns_path = self.settings.workspace_root / "canonical" / "column_inventory.csv"
        if not profile_path.exists() or not columns_path.exists():
            raise RuntimeError("Run audit-data before proposing hypotheses")
        import pandas as pd

        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        columns = pd.read_csv(columns_path).to_dict(orient="records")
        from .research import scientific_role, load_role_overrides

        overrides = load_role_overrides(self.settings)
        for item in columns:
            item["scientific_role"] = scientific_role(str(item["column"]))
            if item["scientific_role"] not in {"identity", "quality"}:
                item["scientific_role"] = overrides.get(
                    str(item["column"]), item["scientific_role"]
                )
        evidence = []
        for kind in (
            "measurement",
            "assignment",
            "mechanism",
            "relationship",
            "limitation",
            "counterexample",
        ):
            rows = self.db.rows(
                """
                SELECT e.evidence_id,e.doc_key,e.page,e.kind,e.claim,e.locator,
                       e.confidence,e.payload_json,d.title,d.doi
                FROM evidence e JOIN documents d USING(doc_key)
                WHERE e.kind=? ORDER BY e.confidence DESC,e.created_at DESC LIMIT 20
                """,
                (kind,),
            )
            for row in rows:
                payload = json.loads(row.pop("payload_json"))
                row["conditions"] = payload.get("conditions", [])
                row["ambiguities"] = payload.get("ambiguities", [])
                row["observables"] = payload.get("observables", [])
                row["directly_visible"] = payload.get("directly_visible")
                row["verification"] = payload.get(
                    "verification", "legacy_model_extracted_unreviewed"
                )
                row["claim"] = row["claim"][:1500]
                evidence.append(row)
        existing = []
        for row in self.db.rows(
            "SELECT spec_json FROM hypotheses ORDER BY created_at DESC LIMIT 50"
        ):
            payload = json.loads(row["spec_json"])
            existing.append(
                {name: payload.get(name) for name in ("name", "symbol", "operation", "inputs")}
            )
        return {
            "profile": profile,
            "columns": columns,
            "evidence": evidence,
            "existing_hypotheses": existing,
        }

    def propose(self, count: int = 8) -> dict[str, Any]:
        if self.llm is None:
            raise RuntimeError("A SiliconFlow client is required to propose hypotheses")
        context = self._context()
        role_by_column = {
            str(item["column"]): str(item.get("scientific_role") or item.get("role") or "")
            for item in context["columns"]
        }
        user_prompt = (
            f"请提出最多 {count} 个候选。下面是机器生成的数据清单和带定位的证据。"
            "先检查变量是否真实存在；如果证据不足，可少提但不能编造。\n\n" + json_dumps(context)
        )
        prompt_hash = sha256_text(self.system_prompt + user_prompt)
        response = self.llm.chat_json(
            [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=self.settings.main_model,
            max_tokens=8000,
            temperature=0.15,
        )
        accepted: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        raw_candidates = response.get("hypotheses", [])
        if not isinstance(raw_candidates, list):
            raise ValueError("Model hypotheses must be a list")
        available_evidence = {
            row["evidence_id"] for row in self.db.rows("SELECT evidence_id FROM evidence")
        }
        reserved_symbols = set(role_by_column)
        for row in self.db.rows("SELECT spec_json FROM hypotheses"):
            reserved_symbols.add(json.loads(row["spec_json"]).get("symbol", ""))
        for raw in raw_candidates[: max(1, min(count, 20))]:
            try:
                spec = DescriptorSpec.model_validate(raw)
                validate_candidate(spec, role_by_column, available_evidence, reserved_symbols)
                payload = spec.model_dump()
                hypothesis_id = self.db.save_hypothesis(
                    payload, prompt_hash, self.settings.main_model
                )
                payload["hypothesis_id"] = hypothesis_id
                accepted.append(payload)
                reserved_symbols.add(spec.symbol)
            except (ValueError, KeyError, TypeError) as exc:
                rejected.append({"candidate": raw, "validation_error": str(exc)})
        return {"accepted": accepted, "rejected": rejected, "prompt_hash": prompt_hash}

    def materialize(
        self, hypothesis_id: str | None = None, table: str | None = None
    ) -> dict[str, Any]:
        import pandas as pd

        from .utils import sha256_file, utc_now

        if table:
            requested = Path(table)
            table_path = (
                requested if requested.is_absolute() else self.settings.project_root / requested
            )
        else:
            table_path = self.settings.workspace_root / "canonical" / "model_table.csv"
        if not table_path.exists():
            raise FileNotFoundError(f"Model table not found: {table_path}")
        destination = self.settings.workspace_root / "canonical" / "model_table_enriched.csv"
        if table_path.resolve() == destination.resolve():
            raise ValueError(
                "Materialize from the original model table, not from the enriched output"
            )
        if hypothesis_id:
            rows = self.db.rows(
                "SELECT hypothesis_id, spec_json FROM hypotheses WHERE hypothesis_id=?",
                (hypothesis_id,),
            )
            if not rows:
                raise ValueError("Unknown hypothesis_id; no output was overwritten")
        else:
            rows = self.db.rows(
                "SELECT hypothesis_id, spec_json FROM hypotheses WHERE status IN ('proposed','calculable')"
            )
        frame = pd.read_csv(table_path)
        inventory_path = self.settings.workspace_root / "canonical" / "column_inventory.csv"
        if not inventory_path.exists():
            raise RuntimeError("Run audit-data before materializing candidates")
        inventory = pd.read_csv(inventory_path)
        roles = dict(
            zip(inventory["column"].astype(str), inventory["role"].astype(str), strict=True)
        )
        from .research import scientific_role, load_role_overrides

        overrides = load_role_overrides(self.settings)
        for column in roles:
            known_role = scientific_role(column)
            if known_role != "unassigned":
                roles[column] = known_role
            if roles[column] not in {"identity", "quality", "spectrum_quality"}:
                roles[column] = overrides.get(column, roles[column])
        available_evidence = {
            item["evidence_id"] for item in self.db.rows("SELECT evidence_id FROM evidence")
        }
        accepted: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        for row in rows:
            try:
                spec = DescriptorSpec.model_validate(json.loads(row["spec_json"]))
                validate_candidate(spec, roles, available_evidence, set(frame.columns))
                computed = DescriptorEngine.compute_table(frame, spec)
                non_null = int(computed.notna().sum())
                if non_null < 10:
                    raise ValueError(f"Only {non_null} finite values; at least 10 are required")
                frame[spec.symbol] = computed
                accepted.append(
                    {
                        "hypothesis_id": row["hypothesis_id"],
                        "column": spec.symbol,
                        "non_null": non_null,
                        "coverage_fraction": non_null / max(1, len(frame)),
                        "operation": spec.operation,
                        "inputs": spec.inputs,
                        "dataset_scope": spec.dataset_scope,
                        "input_bounds": spec.input_bounds,
                        "evidence_level": "numerically_calculable_only_not_mechanism_validated",
                    }
                )
                with self.db.connect() as con:
                    con.execute(
                        "UPDATE hypotheses SET status='calculable', updated_at=? WHERE hypothesis_id=?",
                        (utc_now(), row["hypothesis_id"]),
                    )
            except Exception as exc:
                rejected.append({"hypothesis_id": row["hypothesis_id"], "error": str(exc)})
                with self.db.connect() as con:
                    con.execute(
                        "UPDATE hypotheses SET status='calculation_failed', updated_at=? WHERE hypothesis_id=?",
                        (utc_now(), row["hypothesis_id"]),
                    )
        frame.to_csv(destination, index=False, encoding="utf-8-sig")
        manifest = {
            "spectrum_summary_version": SPECTRUM_SUMMARY_VERSION,
            "source_table": str(table_path),
            "source_sha256": sha256_file(table_path),
            "destination": str(destination),
            "destination_sha256": sha256_file(destination),
            "accepted": accepted,
            "rejected": rejected,
        }
        (destination.with_suffix(".manifest.json")).write_text(
            json_dumps(manifest), encoding="utf-8"
        )
        return manifest
