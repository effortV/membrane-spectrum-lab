from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import shutil
import threading
import time

import httpx
import pymupdf
import numpy as np
import pandas as pd
import pytest
from dotenv import dotenv_values
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from tenacity import wait_none

from xps_agent.config import Settings, save_api_settings
from xps_agent.data import DataAuditor
from xps_agent.db import StateDB
from xps_agent.evidence import EvidenceExtractor, PageEvidence
from xps_agent.features import DescriptorEngine, DescriptorSpec, SpectrumSummary
from xps_agent.hypotheses import HypothesisEngine, validate_candidate
from xps_agent.literature import LiteratureError, LiteratureService, validate_public_url
from xps_agent.llm import LLMBudgetExceeded, LLMError, LLMTimeoutError, SiliconFlowClient
from xps_agent.tasks import BackgroundTasks, OperationStopped
from xps_agent.ml import NestedGroupEvaluator, prepare_evaluation_data
from xps_agent.security import hash_ui_password, safe_error, verify_ui_password
from xps_agent.utils import sha256_text


def spec(**overrides):
    payload = {
        "name": "Peak shape balance",
        "symbol": "shape_balance",
        "operation": "ratio",
        "inputs": ["a", "b"],
        "equation": "a/b",
        "units": "dimensionless",
        "mechanism_chain": ["shape", "heterogeneity", "transport"],
        "applicability": ["same energy window"],
        "confounders": ["baseline"],
        "falsification_tests": ["group holdout", "independent experiment"],
        "expected_direction": "unknown",
        "identifiability": "measured",
        "novelty_queries": ["XPS peak shape membrane transport"],
    }
    payload.update(overrides)
    return DescriptorSpec.model_validate(payload)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setenv("XPS_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("SILICONFLOW_API_KEY", "fake-key-for-tests")
    root = Path(__file__).parents[1]
    shutil.copytree(root / "prompts", tmp_path / "prompts")
    settings = replace(
        Settings.load(tmp_path), siliconflow_api_key="fake-key-for-tests", ui_password_hash=None
    )
    settings.ensure_workspace()
    db = StateDB(settings.database_path)
    db.initialize()
    return settings, db


def mock_llm(settings, db, handler):
    client = SiliconFlowClient(settings, db)
    client.client.close()
    client.client = httpx.Client(
        base_url=settings.siliconflow_base_url, transport=httpx.MockTransport(handler)
    )
    return client


def completion(content='{"ok": true}', finish="stop"):
    return httpx.Response(
        200,
        json={
            "choices": [
                {"message": {"role": "assistant", "content": content}, "finish_reason": finish}
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        },
    )


@pytest.mark.parametrize("operation", ["ratio", "log_ratio", "difference", "normalized_difference"])
def test_pair_operation_cannot_silently_ignore_extra_input(operation):
    with pytest.raises(ValueError):
        spec(operation=operation, inputs=["a", "b", "c"])


def test_denominator_and_positive_log_domain():
    frame = pd.DataFrame({"a": [1, -2, 0], "b": [0, 1, 1]})
    assert pd.isna(DescriptorEngine.compute_table(frame, spec()).iloc[0])
    values = DescriptorEngine.compute_table(frame, spec(operation="log_ratio"))
    assert values.isna().all()


def test_machine_applicability_gate():
    frame = pd.DataFrame({"a": [2, 3, 11], "b": [1, 1, 1], "dataset": ["NF", "RO", "NF"]})
    result = DescriptorEngine.compute_table(
        frame, spec(dataset_scope=["NF"], input_bounds={"a": (0, 10)})
    )
    assert result.iloc[0] == 2
    assert result.iloc[1:].isna().all()


def test_irregular_grid_and_duplicate_points_do_not_create_physics():
    dense = np.linspace(395, 405, 1001)
    uneven = np.unique(np.concatenate([np.linspace(395, 398, 800), np.linspace(398, 405, 400)]))

    def gaussian(x):
        return np.exp(-0.5 * ((x - 400) / 1.1) ** 2)

    reference = SpectrumSummary.compute(dense, gaussian(dense))
    tested = SpectrumSummary.compute(uneven, gaussian(uneven))
    duplicated_x = np.concatenate([dense, dense[:300]])
    duplicated = SpectrumSummary.compute(duplicated_x, gaussian(duplicated_x))
    for name in ("centroid_eV", "spread_eV", "central_80_width_eV", "spectral_entropy"):
        assert tested[name] == pytest.approx(reference[name], abs=0.02)
        assert duplicated[name] == pytest.approx(reference[name], abs=1e-10)


def test_unsafe_symbols_citations_and_quality_inputs_rejected():
    with pytest.raises(ValueError):
        spec(symbol="../target")
    with pytest.raises(ValueError, match="conflicts"):
        validate_candidate(spec(), {"a": "xps", "b": "xps"}, set(), {"shape_balance"})
    with pytest.raises(ValueError, match="Evidence"):
        validate_candidate(spec(evidence_ids=["invented"]), {"a": "xps", "b": "xps"}, set(), set())
    with pytest.raises(ValueError, match="quality"):
        validate_candidate(
            spec(inputs=["N__point_count", "b"]),
            {"N__point_count": "spectrum_shape", "b": "xps"},
            set(),
            set(),
        )


def test_invalid_candidate_does_not_abort_batch(setup, monkeypatch):
    settings, db = setup

    class FakeLLM:
        def chat_json(self, *args, **kwargs):
            return {"hypotheses": [spec(inputs=["missing", "b"]).model_dump(), spec().model_dump()]}

    engine = HypothesisEngine(settings, db, FakeLLM())
    monkeypatch.setattr(
        engine,
        "_context",
        lambda: {
            "columns": [{"column": "a", "role": "xps"}, {"column": "b", "role": "xps"}],
            "evidence": [],
            "profile": {},
        },
    )
    result = engine.propose(2)
    assert len(result["accepted"]) == 1
    assert len(result["rejected"]) == 1


def test_rejected_materialized_column_is_not_exported(setup):
    settings, db = setup
    canonical = settings.workspace_root / "canonical"
    pd.DataFrame({"a": range(9), "b": [1] * 9}).to_csv(canonical / "model_table.csv", index=False)
    pd.DataFrame({"column": ["a", "b"], "role": ["xps", "xps"]}).to_csv(
        canonical / "column_inventory.csv", index=False
    )
    hypothesis = db.save_hypothesis(spec().model_dump(), None, "fake")
    result = HypothesisEngine(settings, db).materialize()
    assert not result["accepted"] and result["rejected"]
    assert "shape_balance" not in pd.read_csv(result["destination"])
    assert (
        db.rows("SELECT status FROM hypotheses WHERE hypothesis_id=?", (hypothesis,))[0]["status"]
        == "calculation_failed"
    )


@pytest.mark.parametrize(
    "baseline,candidates",
    [(["y"], ["c"]), (["a"], ["y"]), (["g"], ["c"]), (["a"], ["a"]), (["a", "a"], ["c"])],
)
def test_ml_rejects_target_group_and_duplicate_predictors(baseline, candidates):
    frame = pd.DataFrame({"y": [1, 2, 3], "g": ["x", "y", "z"], "a": [1, 2, 3], "c": [3, 2, 1]})
    with pytest.raises(ValueError):
        prepare_evaluation_data(frame, "y", "g", baseline, candidates)


def test_ml_cohort_filters_nonfinite_and_unmeasured_candidates():
    frame = pd.DataFrame(
        {
            "y": [1, np.inf, 3, 4, 5],
            "g": ["a", "b", "", "d", "e"],
            "a": [1] * 5,
            "c": [1, 2, 3, np.nan, 5],
        }
    )
    data, report = prepare_evaluation_data(frame, "y", "g", ["a"], ["c"])
    assert data["_input_row_index"].tolist() == [0, 4]
    assert report["dropped_missing_candidates"] == 1
    assert report["dropped_missing_or_nonfinite_target_or_group"] == 2


def test_same_doi_cannot_cross_selected_groups():
    frame = pd.DataFrame(
        {
            "y": [1, 2, 3],
            "g": ["NF:ref:1", "RO:ref:9", "other"],
            "doi_group": ["10.1/same", "10.1/same", "10.1/other"],
            "a": [1] * 3,
            "c": [2] * 3,
        }
    )
    with pytest.raises(ValueError, match="DOI"):
        prepare_evaluation_data(frame, "y", "g", ["a"], ["c"])


def test_nested_group_run_is_auditable_without_heavy_grid(setup, monkeypatch):
    settings, db = setup
    rng = np.random.default_rng(7)
    a, c = rng.normal(size=32), rng.normal(size=32)
    frame = pd.DataFrame(
        {
            "y": a + 2 * c + rng.normal(scale=0.1, size=32),
            "paper": np.repeat(np.arange(8), 4),
            "a": a,
            "c": c,
            "record_id": [f"row-{index}" for index in range(32)],
        }
    )
    table = settings.workspace_root / "synthetic.csv"
    frame.to_csv(table, index=False)
    monkeypatch.setattr(
        NestedGroupEvaluator,
        "_pipeline_and_grid",
        staticmethod(
            lambda seed: (
                Pipeline([("impute", SimpleImputer()), ("model", Ridge())]),
                {"model__alpha": [0.1]},
            )
        ),
    )
    result = NestedGroupEvaluator(settings, db).evaluate(
        table, "y", "paper", ["a"], ["c"], outer_splits=4, inner_splits=2
    )
    splits = json.loads((Path(result["run_dir"]) / "splits.json").read_text())
    for split in splits:
        assert not set(split["train_groups"]) & set(split["test_groups"])
    predictions = pd.read_csv(Path(result["run_dir"]) / "predictions.csv")
    assert len(predictions) == 64
    assert predictions.groupby("feature_set")["input_row_index"].nunique().eq(32).all()
    assert "training_mean_dummy_out_of_fold" in result
    assert "group_balanced_out_of_fold" in result
    assert (Path(result["run_dir"]) / "model_selection.json").exists()


def test_auth_failure_is_not_retried(setup):
    settings, db = setup
    count = []

    def handler(request):
        count.append(request)
        return httpx.Response(401)

    with mock_llm(settings, db, handler) as client:
        with pytest.raises(LLMError, match="not retried"):
            client.chat([{"role": "user", "content": "test"}])
    assert len(count) == 1
    assert db.rows("SELECT COUNT(*) count FROM llm_request_events")[0]["count"] == 1


def test_retry_setting_is_honored(setup, monkeypatch):
    settings, db = setup
    settings = replace(settings, llm_max_retries=2)
    monkeypatch.setattr("xps_agent.llm.wait_exponential_jitter", lambda **kwargs: wait_none())
    count = []

    def handler(request):
        count.append(request)
        return httpx.Response(503)

    with mock_llm(settings, db, handler) as client:
        with pytest.raises(LLMError):
            client.chat([{"role": "user", "content": "test"}])
    assert len(count) == 2


def test_read_timeout_does_not_submit_four_paid_requests(setup):
    settings, db = setup
    seen = []

    def handler(request):
        seen.append(request)
        raise httpx.ReadTimeout("fake-key-for-tests", request=request)

    with mock_llm(settings, db, handler) as client:
        with pytest.raises(LLMTimeoutError, match="不自动重复提交"):
            client.chat([{"role": "user", "content": "test"}], model=settings.vision_model)
    assert len(seen) == 1
    event = db.rows("SELECT error FROM llm_request_events")[0]
    assert "fake-key-for-tests" not in event["error"]


def test_vision_has_short_timeout_without_changing_main_model(setup):
    settings, db = setup
    settings = replace(settings, vision_timeout_seconds=90)
    seen = []

    def handler(request):
        seen.append(request.extensions["timeout"])
        return completion()

    with mock_llm(settings, db, handler) as client:
        client.chat([{"role": "user", "content": "test"}], model=settings.vision_model)
        client.chat([{"role": "user", "content": "test"}])
    assert seen[0]["read"] == 90
    assert seen[0]["connect"] == 10
    assert seen[1]["read"] == settings.llm_timeout_seconds


def test_visual_transient_retries_have_separate_cap(setup, monkeypatch):
    settings, db = setup
    monkeypatch.setattr("xps_agent.llm.wait_exponential_jitter", lambda **kwargs: wait_none())
    seen = []
    with mock_llm(
        settings, db, lambda request: seen.append(request) or httpx.Response(503)
    ) as client:
        with pytest.raises(LLMError):
            client.chat([{"role": "user", "content": "test"}], model=settings.vision_model)
    assert len(seen) == 2


def test_cancel_check_prevents_even_cached_model_call(setup):
    settings, db = setup
    with mock_llm(settings, db, lambda request: completion()) as client:
        client.check_cancelled = lambda: (_ for _ in ()).throw(OperationStopped("stop"))
        with pytest.raises(OperationStopped):
            client.chat([{"role": "user", "content": "test"}])
        assert client.network_requests == 0


def configure_fake_pdf(extractor, monkeypatch, digest="pdf-1"):
    monkeypatch.setattr(
        extractor.pdfs,
        "parse_pdf",
        lambda path: {
            "sha256": digest,
            "pages": [
                {"page": 1, "text": "XPS", "relevance_score": 5},
                {"page": 2, "text": "XPS", "relevance_score": 5},
            ],
        },
    )
    monkeypatch.setattr(
        extractor.pdfs,
        "render_pages",
        lambda path, pages, sha: [Path(f"{page}.png") for page in pages],
    )


class FakePageLLM:
    network_requests = 0

    def __init__(self, fail_page=None):
        self.fail_page = fail_page
        self.calls = []

    def vision_json(self, prompt, images):
        page = int(images[0].stem)
        self.calls.append(page)
        if page == self.fail_page:
            raise LLMTimeoutError("page timeout")
        return {
            "items": [
                {
                    "kind": "measurement",
                    "claim": f"page {page}",
                    "locator": f"Fig {page}",
                    "directly_visible": True,
                }
            ]
        }


def add_fake_document(db, key="doc", status="parsed"):
    db.upsert_document(
        {"doc_key": key, "title": key, "source": "test", "local_path": "fake.pdf", "status": status}
    )


def test_page_failure_is_skipped_and_success_checkpoint_is_resumed(setup, monkeypatch):
    settings, db = setup
    add_fake_document(db)
    first = FakePageLLM(fail_page=2)
    extractor = EvidenceExtractor(settings, db, first)
    configure_fake_pdf(extractor, monkeypatch)
    result = extractor.extract_registered(5)
    assert first.calls == [1, 2]
    assert result["pages_completed"] == 1 and result["pages_failed"] == 1
    assert db.rows("SELECT status FROM documents")[0]["status"] == "evidence_failed"
    second = FakePageLLM()
    extractor = EvidenceExtractor(settings, db, second)
    configure_fake_pdf(extractor, monkeypatch)
    result = extractor.extract_registered(5, "failed")
    assert second.calls == [2]
    assert result["checkpoint_hits"] == 1
    assert len(db.rows("SELECT * FROM evidence")) == 2
    assert db.rows("SELECT status FROM documents")[0]["status"] == "evidence_extracted"


def test_changed_pdf_does_not_reuse_completed_page_checkpoints(setup, monkeypatch):
    settings, db = setup
    add_fake_document(db)
    for digest in ["pdf-1", "pdf-2"]:
        llm = FakePageLLM()
        extractor = EvidenceExtractor(settings, db, llm)
        configure_fake_pdf(extractor, monkeypatch, digest)
        extractor.extract_document({"doc_key": "doc", "local_path": "fake.pdf"})
        assert llm.calls == [1, 2]


def test_normal_batch_never_requeues_failed_articles(setup, monkeypatch):
    settings, db = setup
    add_fake_document(db, "new", "parsed")
    add_fake_document(db, "old-failure", "evidence_failed")
    extractor = EvidenceExtractor(settings, db, FakePageLLM())
    seen = []
    monkeypatch.setattr(
        extractor,
        "extract_document",
        lambda row: seen.append(row["doc_key"]) or {"pages": 0, "evidence": 0},
    )
    extractor.extract_registered()
    assert seen == ["new"]
    seen.clear()
    extractor.extract_registered(mode="failed")
    assert seen == ["old-failure"]
    with pytest.raises(ValueError):
        extractor.extract_registered(mode="injected")


def test_cancelled_article_stays_resumable_and_counts_saved_page(setup, monkeypatch):
    settings, db = setup
    add_fake_document(db)
    llm = FakePageLLM()

    def check():
        if llm.calls:
            raise OperationStopped("stop")

    extractor = EvidenceExtractor(settings, db, llm, check_cancelled=check)
    configure_fake_pdf(extractor, monkeypatch)
    result = extractor.extract_registered()
    assert result["interrupted"] and result["evidence"] == 1
    assert result["pages_completed"] == 1
    assert llm.calls == [1]
    assert db.rows("SELECT status FROM documents")[0]["status"] == "parsed"


def wait_task(manager, owner="owner"):
    end = time.monotonic() + 3
    while time.monotonic() < end:
        snapshot = manager.snapshot(owner)
        if snapshot and snapshot["status"] != "running":
            return snapshot
        threading.Event().wait(0.01)
    raise AssertionError("test worker did not finish")


def test_background_task_is_responsive_cancellable_and_owned(tmp_path):
    gate, state = threading.Lock(), {}
    manager = BackgroundTasks(gate, state)
    entered, release = threading.Event(), threading.Event()

    def worker(context):
        context.update(stage="waiting", page=2)
        entered.set()
        release.wait(2)
        context.check()

    task_id = manager.start("owner", "test", "extraction", worker, journal_root=tmp_path)
    assert entered.wait(1)
    assert manager.snapshot("owner")["progress"]["page"] == 2
    assert not manager.snapshot("another")["is_owner"]
    with pytest.raises(ValueError):
        manager.cancel(task_id, "another")
    with pytest.raises(RuntimeError, match="未重复提交"):
        manager.start("owner", "duplicate", "ask", lambda context: "bad")
    manager.cancel(task_id, "owner")
    release.set()
    assert wait_task(manager)["status"] == "cancelled"
    assert not gate.locked() and not state
    journal = json.loads((tmp_path / f"{task_id}.json").read_text(encoding="utf-8"))
    assert journal["status"] == "cancelled" and "result" not in journal


def test_background_failure_releases_paid_gate_and_redacts_error(monkeypatch):
    monkeypatch.setenv("SILICONFLOW_API_KEY", "test-sensitive-value")
    gate, state = threading.Lock(), {}
    manager = BackgroundTasks(gate, state)

    def worker(context):
        raise RuntimeError("failure test-sensitive-value")

    manager.start("owner", "test", "ask", worker)
    result = wait_task(manager)
    assert result["status"] == "failed" and "test-sensitive-value" not in result["error"]
    assert not gate.locked()


def test_background_deadline_stops_next_action_and_preserves_journal(tmp_path):
    gate, state = threading.Lock(), {}
    manager = BackgroundTasks(gate, state)

    def worker(context):
        threading.Event().wait(0.02)
        context.check()

    manager.start("owner", "test", "ask", worker, timeout=0.001, journal_root=tmp_path)
    assert wait_task(manager)["status"] == "timed_out"
    assert not gate.locked()


def test_journal_failure_does_not_strand_gate_or_share_answers(tmp_path):
    blocked = tmp_path / "not-a-directory"
    blocked.write_text("test")
    gate, state = threading.Lock(), {}
    manager = BackgroundTasks(gate, state)
    manager.start("owner", "test", "ask", lambda context: "private answer", journal_root=blocked)
    assert wait_task(manager)["result"] == "private answer"
    assert not gate.locked()
    assert manager.snapshot("another") is None


def test_request_ledger_cache_and_budget(setup):
    settings, db = setup
    settings = replace(settings, llm_max_requests_per_action=2)
    with mock_llm(settings, db, lambda request: completion()) as client:
        messages = [{"role": "user", "content": "test"}]
        client.chat(messages)
        client.chat(messages)
        assert client.network_requests == 1
        client.chat(messages, use_cache=False)
        with pytest.raises(LLMBudgetExceeded):
            client.chat([{"role": "user", "content": "new"}], use_cache=False)
    statuses = db.rows("SELECT status,COUNT(*) count FROM llm_request_events GROUP BY status")
    assert {row["status"]: row["count"] for row in statuses} == {"cache_hit": 1, "succeeded": 2}


def test_truncated_json_is_not_accepted_or_retried(setup):
    settings, db = setup
    with mock_llm(
        settings, db, lambda request: completion('{"partial":', finish="length")
    ) as client:
        with pytest.raises(LLMError, match="截断"):
            client.chat_json([{"role": "user", "content": "test"}])
        assert client.network_requests == 1


def test_evidence_confidence_must_be_bounded():
    with pytest.raises(ValueError):
        PageEvidence.model_validate(
            {
                "items": [
                    {
                        "kind": "measurement",
                        "claim": "test",
                        "locator": "Fig. 1",
                        "confidence": 9,
                        "directly_visible": True,
                    }
                ]
            }
        )


def test_zero_relevant_pages_are_not_labeled_extracted(setup, monkeypatch):
    settings, db = setup
    db.upsert_document(
        {
            "doc_key": "test",
            "title": "test",
            "source": "test",
            "local_path": "dummy.pdf",
            "status": "parsed",
        }
    )
    extractor = EvidenceExtractor(settings, db, object())
    monkeypatch.setattr(extractor.pdfs, "parse_pdf", lambda path: {"sha256": "test", "pages": []})
    result = extractor.extract_document({"doc_key": "test", "local_path": "dummy.pdf"})
    assert result == {"pages": 0, "evidence": 0}
    assert db.rows("SELECT status FROM documents")[0]["status"] == "evidence_no_relevant_pages"


def test_duplicate_environment_entries_do_not_restore_old_secret(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENALEX_API_KEY", "old")
    (tmp_path / ".env").write_text(
        "OPENALEX_API_KEY=old\nOPENALEX_API_KEY=older\nXPS_MAIN_MODEL=preserved\n"
    )
    save_api_settings(tmp_path, {"OPENALEX_API_KEY": "new"})
    assert dotenv_values(tmp_path / ".env")["OPENALEX_API_KEY"] == "new"
    assert (tmp_path / ".env").read_text().count("OPENALEX_API_KEY=") == 1
    save_api_settings(tmp_path, {"OPENALEX_API_KEY": ""})
    assert dotenv_values(tmp_path / ".env")["OPENALEX_API_KEY"] == ""


def test_secret_error_redaction(monkeypatch):
    monkeypatch.setenv("OPENALEX_API_KEY", "supersecret")
    assert "supersecret" not in safe_error("Request https://api.test?api_key=supersecret failed")
    assert "unknownsecret" not in safe_error("Authorization=unknownsecret")


def test_password_hashing():
    password = "test-password-123"
    first, second = hash_ui_password(password), hash_ui_password(password)
    assert first != second and password not in first
    assert verify_ui_password(password, first)
    assert not verify_ui_password("incorrect", first)


def test_private_download_url_is_rejected():
    with pytest.raises(LiteratureError, match="Private"):
        validate_public_url("http://127.0.0.1/private.pdf")


def test_partial_range_is_not_mistaken_for_complete_pdf(tmp_path):
    partial = tmp_path / "test.part"
    partial.write_bytes(b"%PDF-not-complete")
    response = httpx.Response(416, headers={"Content-Range": f"bytes */{partial.stat().st_size}"})
    with pytest.raises(LiteratureError, match="incomplete"):
        LiteratureService._save_pdf_response(
            response, partial, tmp_path / "test.pdf", partial.stat().st_size
        )
    assert not (tmp_path / "test.pdf").exists()


def test_source_row_provenance_retains_blank_rows(setup):
    settings, db = setup
    path = settings.project_root / "simple.xlsx"
    pd.DataFrame({"Ref": [1, np.nan, 2], "a": [2, np.nan, 3]}).to_excel(path, index=False)
    item = DataAuditor(settings, db).read_sheet(path, "Sheet1")
    assert item.source_rows == [2, 4]
    assert item.frame["Ref"].tolist() == [1, 2]


def test_entitlement_headers_are_not_forwarded_to_pdf_cdn(setup, monkeypatch):
    settings, db = setup
    document = pymupdf.open()
    document.new_page().insert_text((20, 20), "Test PDF")
    pdf_bytes = document.tobytes()
    document.close()
    seen = []

    def handler(request):
        seen.append(request)
        if request.url.host == "api.elsevier.test":
            return httpx.Response(302, headers={"Location": "https://cdn.example.test/test.pdf"})
        return httpx.Response(200, content=pdf_bytes)

    monkeypatch.setattr("xps_agent.literature.validate_public_url", lambda url: None)
    service = LiteratureService(settings, db)
    service.client.close()
    service.client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)
    try:
        result = service._try_url(
            "https://api.elsevier.test/content",
            settings.workspace_root / "library" / "test.pdf",
            {"X-ELS-APIKey": "not-a-real-key", "X-ELS-Insttoken": "test-token"},
        )
        assert result.exists()
        assert len(seen) == 2
        assert "X-ELS-APIKey" in seen[0].headers
        assert "X-ELS-APIKey" not in seen[1].headers
        assert "X-ELS-Insttoken" not in seen[1].headers
    finally:
        service.client.close()


def test_resume_files_are_source_specific(setup, monkeypatch):
    settings, db = setup
    url1, url2 = "https://one.example.test/a.pdf", "https://two.example.test/a.pdf"
    destination = settings.workspace_root / "library" / "test.pdf"
    partial = destination.with_suffix(f".{sha256_text(url1)[:12]}.part")
    partial.parent.mkdir(parents=True, exist_ok=True)
    partial.write_bytes(b"%PDF-source-one-partial")
    document = pymupdf.open()
    document.new_page()
    pdf_bytes = document.tobytes()
    document.close()
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, content=pdf_bytes)

    monkeypatch.setattr("xps_agent.literature.validate_public_url", lambda url: None)
    service = LiteratureService(settings, db)
    service.client.close()
    service.client = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        service._try_url(url2, destination)
        assert "range" not in seen[0].headers
        assert partial.exists()
    finally:
        service.client.close()


def test_materialization_does_not_overwrite_unknown_or_enriched_source(setup):
    settings, db = setup
    canonical = settings.workspace_root / "canonical"
    table = canonical / "model_table.csv"
    pd.DataFrame({"a": [1], "b": [2]}).to_csv(table, index=False)
    with pytest.raises(ValueError, match="Unknown"):
        HypothesisEngine(settings, db).materialize("not-a-hypothesis")
    enriched = canonical / "model_table_enriched.csv"
    enriched.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError, match="original model table"):
        HypothesisEngine(settings, db).materialize(table=str(enriched))
    assert enriched.read_text() == "a,b\n1,2\n"


def test_enriched_stale_manifest_is_rejected_in_backend(setup):
    settings, db = setup
    canonical = settings.workspace_root / "canonical"
    table = canonical / "model_table_enriched.csv"
    pd.DataFrame({"y": [1, 2, 3], "g": [1, 2, 3], "a": [1, 1, 1], "c": [2, 2, 2]}).to_csv(
        table, index=False
    )
    with pytest.raises(ValueError, match="manifest"):
        NestedGroupEvaluator(settings, db).evaluate(table, "y", "g", ["a"], ["c"])


def test_dataset_scope_keeps_original_source_positions():
    frame = pd.DataFrame(
        {"y": [1, 2, 3], "g": [1, 2, 3], "a": [1] * 3, "c": [2] * 3, "dataset": ["NF", "RO", "NF"]}
    )
    data, report = prepare_evaluation_data(frame, "y", "g", ["a"], ["c"], dataset_scope=["NF"])
    assert data["_input_row_index"].tolist() == [0, 2]
    assert report["dropped_outside_dataset_scope"] == 1


def test_cli_errors_exit_without_raw_credential_traceback(monkeypatch, capsys):
    from xps_agent import cli

    monkeypatch.setenv("OPENALEX_API_KEY", "test-secret")

    def fail():
        raise LiteratureError("GET https://api.example?api_key=test-secret failed")

    monkeypatch.setattr(cli, "app", fail)
    with pytest.raises(SystemExit) as result:
        cli.main()
    assert result.value.code == 2
    assert "test-secret" not in capsys.readouterr().out


def test_paid_operation_gate_releases_after_failure(monkeypatch):
    import threading
    from xps_agent import dashboard

    lock, state = threading.Lock(), {}
    monkeypatch.setattr(dashboard, "get_paid_gate", lambda: (lock, state))
    with dashboard.paid_operation("test"):
        with pytest.raises(RuntimeError, match="未重复提交"):
            with dashboard.paid_operation("duplicate"):
                raise AssertionError("Duplicate paid operation should never start")
    assert not lock.locked() and not state
    with pytest.raises(ValueError):
        with dashboard.paid_operation("failure"):
            raise ValueError("test")
    assert not lock.locked() and not state
