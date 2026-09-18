from __future__ import annotations

import platform
import sys
import uuid
from pathlib import Path

import typer
from rich.console import Console
from . import __version__
from .utils import json_dumps

from .agent import ResearchAgent
from .config import Settings
from .data import DataAuditor
from .db import StateDB
from .diagnostics import health_report
from .evidence import EvidenceExtractor
from .hypotheses import HypothesisEngine
from .literature import LiteratureError, LiteratureService
from .llm import LLMError, SiliconFlowClient
from .ml import NestedGroupEvaluator
from .pdfs import PDFService
from .security import safe_error


app = typer.Typer(
    name="xps-agent",
    help="Evidence-grounded XPS + mechanism + ML discovery agent",
    no_args_is_help=True,
)
console = Console()


def context() -> tuple[Settings, StateDB]:
    settings = Settings.load()
    settings.ensure_workspace()
    db = StateDB(settings.database_path)
    db.initialize()
    return settings, db


def show_json(value: object) -> None:
    console.print_json(data=value)


@app.command("doctor")
def doctor() -> None:
    """Check paths, dependencies, configuration, and data availability without exposing secrets."""
    settings, db = context()
    summary = settings.public_summary()
    summary.update(
        {
            "python": sys.version.split()[0],
            "agent_version": __version__,
            "platform": platform.platform(),
            "nf_xlsx": (settings.data_root / "NF.xlsx").exists(),
            "ro_xlsx": (settings.data_root / "RO.xlsx").exists(),
            "reference_pdf_count": len(list(settings.reference_root.rglob("*.pdf")))
            if settings.reference_root.exists()
            else 0,
            "database_ready": db.path.exists(),
            "legacy_final_read_only_policy": True,
        }
    )
    show_json(summary)


@app.command("init-db")
def init_db() -> None:
    """Initialize the SQLite evidence, hypothesis, and run ledger."""
    _, db = context()
    console.print(f"Initialized: {db.path}")


@app.command("health")
def health(
    save_report: bool = typer.Option(
        False, help="Save a unique sanitized JSON under workspace/reports"
    ),
) -> None:
    """Read-only scientific/runtime checks: no API request or paid inference."""
    settings, db = context()
    report = health_report(settings, db)
    if save_report:
        reports = settings.workspace_root / "reports"
        reports.mkdir(parents=True, exist_ok=True)
        destination = reports / f"health_{uuid.uuid4().hex[:12]}.json"
        report["report_path"] = str(destination)
        with destination.open("x", encoding="utf-8") as handle:
            handle.write(json_dumps(report))
    show_json(report)


@app.command("audit-data")
def audit_data(
    export_tables: bool = typer.Option(True, help="Export source sheets as provenance CSVs"),
) -> None:
    """Audit original NF/RO tables and extracted N/O spectra without importing legacy ML results."""
    settings, db = context()
    auditor = DataAuditor(settings, db)
    result = auditor.run()
    written = auditor.export_authoritative_tables() if export_tables else []
    result["exported_source_tables"] = [str(path) for path in written]
    show_json(result)


@app.command("index-library")
def index_library() -> None:
    """Index local reference, downloaded, and manually uploaded PDFs by DOI and SHA-256."""
    settings, db = context()
    service = LiteratureService(settings, db)
    result = service.index_local_pdfs(
        [
            settings.reference_root,
            settings.workspace_root / "library",
            settings.workspace_root / "inbox",
        ]
    )
    show_json(result)


@app.command("literature-search")
def literature_search(
    query: str,
    max_results: int = typer.Option(50, min=1, max=100),
) -> None:
    """Search OpenAlex metadata/OA locations and queue results."""
    settings, db = context()
    keys = LiteratureService(settings, db).search_openalex(query, max_results)
    show_json({"query": query, "queued": len(keys)})


@app.command("literature-fetch")
def literature_fetch(limit: int = typer.Option(50, min=1, max=500)) -> None:
    """Download directly accessible PDFs; export inaccessible entries for manual upload."""
    settings, db = context()
    result = LiteratureService(settings, db).fetch_queued(limit)
    show_json(result)


@app.command("literature-parse")
def literature_parse(limit: int = typer.Option(100, min=1, max=1000)) -> None:
    """Extract local PDF text and score XPS-relevant pages."""
    settings, db = context()
    show_json(PDFService(settings, db).parse_registered(limit))


@app.command("extract-evidence")
def extract_evidence(
    limit: int = typer.Option(20, min=1, max=100),
    mode: str = typer.Option(
        "pending", help="pending / failed / all; failed retries require an explicit action"
    ),
) -> None:
    """Use the lower-cost vision model on selected figure/table pages only."""
    settings, db = context()
    with SiliconFlowClient(settings, db) as llm:
        show_json(EvidenceExtractor(settings, db, llm).extract_registered(limit, mode))


@app.command("evidence-plan")
def evidence_plan(limit: int = typer.Option(1000, min=1, max=5000)) -> None:
    """Preview how many selected pages a VLM evidence run would send before spending money."""
    settings, db = context()
    show_json(PDFService(settings, db).evidence_plan(limit))


@app.command("propose")
def propose(count: int = typer.Option(8, min=1, max=20)) -> None:
    """Use DeepSeek-V4-Pro to propose constrained, falsifiable descriptor hypotheses."""
    settings, db = context()
    llm = SiliconFlowClient(settings, db)
    show_json(HypothesisEngine(settings, db, llm).propose(count))


@app.command("hypotheses")
def hypotheses(limit: int = typer.Option(50, min=1, max=500)) -> None:
    """List stored hypotheses."""
    _, db = context()
    rows = db.rows(
        "SELECT hypothesis_id, name, status, model, created_at FROM hypotheses ORDER BY created_at DESC LIMIT ?",
        (limit,),
    )
    show_json(rows)


@app.command("materialize")
def materialize(
    hypothesis_id: str | None = typer.Option(None, help="One hypothesis; omit for all proposed"),
    table: str | None = typer.Option(
        None, help="Input table; defaults to canonical model_table.csv"
    ),
) -> None:
    """Compute schema-valid descriptor columns with the safe DSL; no arbitrary model code is run."""
    settings, db = context()
    # No API request occurs here; deterministic materialization does not create an LLM client.
    engine = HypothesisEngine(settings, db)
    show_json(engine.materialize(hypothesis_id, table))


@app.command("status")
def status() -> None:
    """Show document, hypothesis, evidence, and experiment counts."""
    _, db = context()
    counts = {
        "documents": db.rows("SELECT status, COUNT(*) AS count FROM documents GROUP BY status"),
        "evidence": db.rows("SELECT kind, COUNT(*) AS count FROM evidence GROUP BY kind"),
        "hypotheses": db.rows("SELECT status, COUNT(*) AS count FROM hypotheses GROUP BY status"),
        "experiments": db.rows(
            "SELECT decision, COUNT(*) AS count FROM experiments GROUP BY decision"
        ),
        "llm_historical_cache_not_billing": db.rows(
            """
            SELECT model, COUNT(*) AS unique_cached_requests, SUM(input_tokens) AS input_tokens,
                   SUM(output_tokens) AS output_tokens
            FROM llm_calls GROUP BY model
            """
        ),
        "llm_network_events_since_upgrade": db.rows(
            "SELECT model,status,COUNT(*) requests,SUM(input_tokens) input_tokens,SUM(output_tokens) output_tokens FROM llm_request_events GROUP BY model,status"
        ),
    }
    show_json(counts)


@app.command("evaluate")
def evaluate(
    table: Path,
    target: str = typer.Option(..., help="Outcome column"),
    group: str = typer.Option(..., help="DOI/article grouping column"),
    baseline: str = typer.Option(..., help="Comma-separated baseline feature columns"),
    candidate: str = typer.Option(..., help="Comma-separated candidate descriptor columns"),
    outer_splits: int = typer.Option(5, min=3, max=10),
    inner_splits: int = typer.Option(4, min=2, max=8),
    hypothesis_id: str | None = typer.Option(None),
    candidate_policy: str = typer.Option("measured_only", help="measured_only (default) or impute"),
    include_unresolved_groups: bool = typer.Option(
        False, help="Allow unknown DOI fallback groups; exploratory only"
    ),
    datasets: str | None = typer.Option(
        None, help="Optional comma-separated dataset scope, e.g. NF or RO"
    ),
) -> None:
    """Compare baseline vs baseline+candidate using identical nested grouped folds."""
    settings, db = context()
    baseline_columns = [item.strip() for item in baseline.split(",") if item.strip()]
    candidate_columns = [item.strip() for item in candidate.split(",") if item.strip()]
    result = NestedGroupEvaluator(settings, db).evaluate(
        table,
        target,
        group,
        baseline_columns,
        candidate_columns,
        outer_splits,
        inner_splits,
        hypothesis_id=hypothesis_id,
        candidate_policy=candidate_policy,
        include_unresolved_groups=include_unresolved_groups,
        dataset_scope=[item.strip() for item in datasets.split(",") if item.strip()]
        if datasets is not None
        else None,
    )
    show_json(result)


@app.command("ask")
def ask(question: str) -> None:
    """Ask the tool-using research agent about data, evidence, hypotheses, or runs."""
    settings, db = context()
    llm = SiliconFlowClient(settings, db)
    console.print(ResearchAgent(settings, db, llm).ask(question))


@app.command("prepare")
def prepare(parse_limit: int = typer.Option(500, min=1, max=5000)) -> None:
    """Run the no-LLM preparation stage: audit data, index local PDFs, and parse them."""
    settings, db = context()
    literature = LiteratureService(settings, db)
    indexed = literature.index_local_pdfs(
        [
            settings.reference_root,
            settings.workspace_root / "library",
            settings.workspace_root / "inbox",
        ]
    )
    auditor = DataAuditor(settings, db)
    profile = auditor.run()
    auditor.export_authoritative_tables()
    parsed = PDFService(settings, db).parse_registered(parse_limit)
    show_json({"profile": profile, "indexed": indexed, "parsed": parsed})


def main() -> None:
    try:
        app()
    except (LLMError, LiteratureError, ValueError, KeyError, OSError) as exc:
        console.print(
            f"Configuration/operation error: {safe_error(exc)}", style="red", markup=False
        )
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
