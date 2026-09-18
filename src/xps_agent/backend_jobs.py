"""Server-only workers. No Streamlit, request retries or client-supplied paths."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Parameters(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Empty(Parameters):
    pass


class Limit(Parameters):
    limit: int = Field(default=5, ge=1, le=500)


class Extraction(Limit):
    limit: int = Field(default=5, ge=1, le=20)
    mode: Literal["pending", "retry_failed"] = "pending"


class Proposal(Parameters):
    limit: int = Field(default=6, ge=1, le=20)


class Search(Parameters):
    query: str = Field(min_length=3, max_length=500)
    limit: int = Field(default=30, ge=1, le=100)


class Fetch(Limit):
    doc_keys: list[str] = Field(min_length=1, max_length=100)


class Relationship(Parameters):
    task: Literal[
        "preparation_to_xps",
        "preparation_xps_to_structure",
        "xps_to_structure",
        "xps_to_performance",
        "structure_to_performance",
        "chain_to_performance",
    ]
    target: str = Field(min_length=1, max_length=500)
    groups: dict[str, list[str]]
    controls: list[str] = Field(default_factory=list, max_length=30)
    datasets: list[Literal["NF", "RO"]] = Field(min_length=1, max_length=2)
    outer_splits: int = Field(default=3, ge=3, le=10)
    inner_splits: int = Field(default=2, ge=2, le=8)
    models: list[Literal["ridge", "random_forest", "extra_trees"]] = Field(
        default=["ridge"], min_length=1, max_length=3
    )
    allow_unresolved: bool = False
    measured_intermediates: bool = True


class ImageAnalysis(Parameters):
    artifact_ids: list[str] = Field(min_length=1, max_length=4)
    context: dict[str, str]


JOB_SPECS = {
    "index": (Empty, "索引本地文献"),
    "parse": (Limit, "读取 PDF 正文"),
    "plan": (Empty, "筛选 XPS 图页"),
    "prepare": (Empty, "整理并读取文献"),
    "search": (Search, "检索 OpenAlex"),
    "fetch": (Fetch, "下载所选文献"),
    "extraction": (Extraction, "视觉证据抽取"),
    "proposal": (Proposal, "提出新描述符"),
    "materialize": (Empty, "计算候选描述符"),
    "data_audit": (Empty, "审计 NF / RO 原始数据"),
    "relationship": (Relationship, "关系研究 · 分组验证"),
    "xps_image": (ImageAnalysis, "分析上传的 XPS 图"),
}
PAID_JOBS = frozenset({"extraction", "proposal", "xps_image"})


def build_worker(settings, db, kind: str, parameters: Parameters):
    def worker(context):
        context.check()
        context.update(stage=JOB_SPECS[kind][1])
        if kind == "data_audit":
            from .data import DataAuditor

            return DataAuditor(settings, db).run()
        if kind == "materialize":
            from .hypotheses import HypothesisEngine

            return HypothesisEngine(settings, db).materialize()
        if kind == "relationship":
            from .research import RelationshipEvaluator, current_research_table

            values = parameters.model_dump()
            task, target = values.pop("task"), values.pop("target")
            groups, controls = values.pop("groups"), values.pop("controls")
            return RelationshipEvaluator(
                settings, db, progress=context.update, check_cancelled=context.check
            ).evaluate(current_research_table(settings), task, target, groups, controls, **values)
        if kind in PAID_JOBS:
            from .llm import SiliconFlowClient

            with SiliconFlowClient(
                settings, db, progress=context.update, check_cancelled=context.check
            ) as llm:
                if kind == "extraction":
                    from .evidence import EvidenceExtractor

                    return EvidenceExtractor(
                        settings, db, llm, progress=context.update, check_cancelled=context.check
                    ).extract_registered(parameters.limit, parameters.mode)
                if kind == "proposal":
                    from .hypotheses import HypothesisEngine

                    return HypothesisEngine(settings, db, llm).propose(parameters.limit)
                from .membrane import MembraneAnalyzer

                images = []
                for key in parameters.artifact_ids:
                    rows = db.rows(
                        "SELECT metadata_json FROM artifacts WHERE artifact_id=? AND kind='xps_uploaded_image'",
                        (key,),
                    )
                    if not rows:
                        raise ValueError("上传图片不存在，请重新上传。")
                    images.append(json.loads(rows[0]["metadata_json"]))
                return MembraneAnalyzer(
                    settings, db, llm, progress=context.update, check_cancelled=context.check
                ).analyze(images, parameters.context)
        from .literature import LiteratureService
        from .pdfs import PDFService

        pdfs = PDFService(settings, db, progress=context.update, check_cancelled=context.check)
        service = LiteratureService(
            settings, db, progress=context.update, check_cancelled=context.check
        )
        roots = [
            settings.reference_root,
            settings.workspace_root / "library",
            settings.workspace_root / "inbox",
        ]
        try:
            if kind == "search":
                service.search_openalex(parameters.query, parameters.limit)
                return service.activity.summary(service.last_batch_id)
            if kind == "fetch":
                return service.fetch_queued(parameters.limit, parameters.doc_keys)
            if kind == "index":
                return service.index_local_pdfs(roots)
            if kind == "parse":
                return pdfs.parse_registered(parameters.limit)
            if kind == "plan":
                return pdfs.evidence_plan(5000)
            return {
                "index": service.index_local_pdfs(roots),
                "text": pdfs.parse_registered(500),
                "plan": pdfs.evidence_plan(5000),
            }
        finally:
            service.client.close()

    return worker


def confined_file(path: str | Path, roots: list[Path], suffix: str) -> Path:
    resolved = Path(path).resolve()
    if (
        resolved.suffix.lower() != suffix
        or not any(resolved.is_relative_to(root.resolve()) for root in roots)
        or not resolved.is_file()
    ):
        raise ValueError("文件不在允许的数据目录，或不存在。")
    return resolved
