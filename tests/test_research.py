from dataclasses import replace
from io import BytesIO
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from PIL import Image
import pytest

from xps_agent.config import Settings
from xps_agent.db import StateDB
from xps_agent.membrane import MembraneAnalyzer, save_xps_image, measured_context
from xps_agent.research import (
    RelationshipEvaluator,
    scientific_role,
    field_catalog,
    prepare_research_data,
)
from xps_agent.storage import migrate_storage
from xps_agent.pdfs import PDFService
from xps_agent.tasks import OperationStopped
from xps_agent.hypotheses import validate_candidate


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("XPS_PROJECT_ROOT", str(tmp_path))
    settings = replace(
        Settings.load(tmp_path),
        storage_root=tmp_path / "central",
        workspace_root=tmp_path / "central" / "workspace",
        database_path=tmp_path / "central" / "workspace" / "state" / "xps_agent.sqlite3",
        siliconflow_api_key="fake-test-key",
    )
    settings.ensure_workspace()
    db = StateDB(settings.database_path)
    db.initialize()
    return settings, db


@pytest.mark.parametrize(
    "column,role",
    [
        ("制备工艺 | 基膜 | 截留分子量", "preparation"),
        ("制备工艺 | 水相单体种类", "preparation"),
        ("分离层/膜结构参数 | MWCO", "structure"),
        ("分离层/膜结构参数 | Zeta电位mV (pH=7)", "structure"),
        ("膜性能参数 | R(NaCl)", "performance"),
        ("操作参数 | 浓度（ppm）", "test_condition"),
        ("XPS参数 | HABD计算", "structure_proxy"),
        ("XPS参数 | C(%)", "xps"),
        ("N__point_count", "quality"),
    ],
)
def test_hierarchical_scientific_roles(column, role):
    assert scientific_role(column) == role


def synthetic():
    x = np.linspace(0, 1, 36)
    return pd.DataFrame(
        {
            "record_id": [f"r{i}" for i in range(36)],
            "dataset": "NF",
            "doi_group": [f"10.test/paper{i // 6}" for i in range(36)],
            "制备工艺 | 单体": ["PIP", "MPD"] * 18,
            "制备工艺 | 反应时间": x * 120,
            "XPS参数 | N(%)": x * 4 + 10,
            "分离层/膜结构参数 | 厚度": x * 10 + 20,
            "膜性能参数 | R(NaCl)": x * 20 + 70,
            "操作参数 | 压力": 5.0,
            "XPS参数 | HABD计算": x * 3,
        }
    )


def test_proxy_is_not_independent_structure_target():
    frame = synthetic()
    with pytest.raises(ValueError, match="target"):
        prepare_research_data(
            frame,
            field_catalog(frame),
            "xps_to_structure",
            "XPS参数 | HABD计算",
            {"xps": ["XPS参数 | N(%)"]},
            [],
            datasets=["NF"],
        )


def test_known_crosslink_proxy_cannot_seed_new_descriptor():
    candidate = SimpleNamespace(
        inputs=["XPS参数 | DNC", "N__spectral_entropy"],
        symbol="new_index",
        evidence_ids=[],
        novelty_queries=["novelty"],
    )
    with pytest.raises(ValueError, match="forbidden"):
        validate_candidate(
            candidate,
            {"XPS参数 | DNC": "xps", "N__spectral_entropy": "spectrum_shape"},
            set(),
            set(),
        )


def test_controls_and_cohort_are_identical_across_feature_sets():
    frame = synthetic()
    frame.loc[0, "XPS参数 | N(%)"] = np.nan
    frame.loc[1, "doi_group"] = np.nan
    data, sets, report = prepare_research_data(
        frame,
        field_catalog(frame),
        "preparation_xps_to_structure",
        "分离层/膜结构参数 | 厚度",
        {"preparation": ["制备工艺 | 单体"], "xps": ["XPS参数 | N(%)"]},
        ["操作参数 | 压力"],
        datasets=["NF"],
    )
    assert len(data) == 34
    assert all("操作参数 | 压力" in values for values in sets.values())
    assert report["missing_measured_intermediates"] == 1
    assert set(data["制备工艺 | 单体"]) == {"PIP", "MPD"}
    assert set(data["_input_row_index"]) == set(range(2, 36))


def test_supplement_url_does_not_become_an_independent_article():
    frame = synthetic()
    frame.loc[:5, "doi_group"] = "10.1073/pnas.2019891118"
    frame.loc[0, "doi_group"] += "/-/dcsupplemental"
    data, _, report = prepare_research_data(
        frame,
        field_catalog(frame),
        "xps_to_structure",
        "分离层/膜结构参数 | 厚度",
        {"xps": ["XPS参数 | N(%)"]},
        [],
        datasets=["NF"],
    )
    assert report["groups"] == 6
    assert data.iloc[0]["_group"] == data.iloc[1]["_group"]
    assert data.iloc[0]["doi_group"].endswith("dcsupplemental")


def test_candidate_ablation_uses_same_measured_cohort():
    frame = synthetic()
    frame["new_XPS_descriptor"] = frame["XPS参数 | N(%)"] ** 2
    frame.loc[0, "new_XPS_descriptor"] = np.nan
    data, sets, report = prepare_research_data(
        frame,
        field_catalog(frame, candidate_symbols=["new_XPS_descriptor"]),
        "xps_to_structure",
        "分离层/膜结构参数 | 厚度",
        {"xps": ["XPS参数 | N(%)", "new_XPS_descriptor"]},
        [],
        datasets=["NF"],
    )
    assert len(data) == report["rows_scored"] == 35
    assert sets["xps_observables"] == ["XPS参数 | N(%)"]
    assert sets["xps_candidate_descriptors"] == ["new_XPS_descriptor"]


def test_relationship_nested_run_supports_categories_and_audit(env):
    settings, db = env
    table = settings.workspace_root / "synthetic.csv"
    synthetic().to_csv(table, index=False)
    progress = []
    result = RelationshipEvaluator(
        settings, db, progress=lambda **data: progress.append(data)
    ).evaluate(
        table,
        "preparation_xps_to_structure",
        "分离层/膜结构参数 | 厚度",
        {"preparation": ["制备工艺 | 单体", "制备工艺 | 反应时间"], "xps": ["XPS参数 | N(%)"]},
        [],
        datasets=["NF"],
        models=["ridge"],
    )
    directory = Path(result["run_dir"])
    predictions = pd.read_csv(directory / "predictions.csv")
    assert set(predictions["feature_set"]) == {"training_mean", "preparation", "xps", "combined"}
    for split in json.loads((directory / "splits.json").read_text()):
        assert not set(split["train_groups"]) & set(split["test_groups"])
    assert any(item.get("fits_finished", 0) > 0 for item in progress)
    assert (
        db.rows("SELECT decision FROM experiments")[0]["decision"]
        == "relationship_requires_external_validation"
    )
    assert not (settings.project_root / "workspace" / "runs").exists()


def image_bytes():
    output = BytesIO()
    Image.new("RGB", (128, 128), "white").save(output, format="PNG")
    return output.getvalue()


class FakeModels:
    def __init__(self, evidence_ids=None):
        self.evidence_ids = evidence_ids or []

    def vision_json(self, prompt, paths, **kwargs):
        return {
            "observations": [
                {
                    "spectrum": "N1s",
                    "visible": ["N1s 坐标可见"],
                    "readable_values": [],
                    "ambiguities": ["无峰面积表"],
                }
            ],
            "limitations": ["未测量性能"],
        }

    def chat_json(self, messages, **kwargs):
        assert "fake-test-key" not in json.dumps(messages)
        return {
            "summary": "不能仅凭此图判断膜性能。",
            "conditional_structure": ["需要拟合和材料信息。"],
            "missing_measurements": ["测量通量、截留率。"],
            "evidence_ids": self.evidence_ids,
        }


def test_xps_upload_saves_original_without_path_traversal(env):
    settings, _ = env
    content = image_bytes()
    image = save_xps_image(settings, "../../escape.png", content)
    assert Path(image["original"]).is_relative_to(settings.upload_root)
    assert Path(image["original"]).read_bytes() == content
    assert image["name"] == "escape.png"
    with pytest.raises(ValueError):
        save_xps_image(settings, "fake.png", b"not an image")


def test_xps_membrane_report_is_conditional_and_not_training_data(env):
    settings, db = env
    image = save_xps_image(settings, "test.png", image_bytes())
    result = MembraneAnalyzer(settings, db, FakeModels()).analyze([image], {"membrane_type": "NF"})
    assert result["evidence"] == []
    assert Path(result["run_dir"], "report.md").exists()
    assert db.rows("SELECT kind FROM artifacts")[0]["kind"] == "membrane_analysis"
    assert not (settings.workspace_root / "canonical" / "model_table.csv").exists()


def test_table_background_is_family_specific_and_not_a_prediction(env):
    settings, _ = env
    table = settings.workspace_root / "canonical" / "model_table.csv"
    original = synthetic()
    other = synthetic().assign(dataset="RO")
    pd.concat([original, other]).to_csv(table, index=False)
    context = measured_context(settings, "NF")
    assert context["datasets"] == ["NF"] and context["rows"] == 36
    assert context["source_sha256"]
    assert "NOT a match" in context["boundary"]
    assert context["variables"]


def test_unknown_membrane_citations_not_published(env):
    settings, db = env
    image = save_xps_image(settings, "test.png", image_bytes())
    with pytest.raises(ValueError, match="证据"):
        MembraneAnalyzer(settings, db, FakeModels(["invented"])).analyze([image], {})
    assert not db.rows("SELECT * FROM artifacts")


def test_pdf_progress_and_cancellation(env, monkeypatch):
    settings, db = env
    for index in range(3):
        db.upsert_document(
            {
                "doc_key": str(index),
                "title": str(index),
                "source": "test",
                "local_path": f"{index}.pdf",
                "status": "indexed",
            }
        )
    updates = []
    reader = PDFService(settings, db, progress=lambda **item: updates.append(item))
    monkeypatch.setattr(reader, "parse_pdf", lambda path: {})
    assert reader.parse_registered()["parsed"] == 3
    assert updates[-1]["documents_finished"] == 3
    with db.connect() as con:
        con.execute("UPDATE documents SET status='indexed'")

    def stop():
        raise OperationStopped("stopped")

    reader.check = stop
    with pytest.raises(OperationStopped):
        reader.parse_registered()
    assert not db.rows("SELECT * FROM documents WHERE status='parse_failed'")


def test_migration_retains_original_and_evidence_ids(tmp_path, monkeypatch):
    project, source, destination = (
        tmp_path / "project",
        tmp_path / "XPS-914",
        tmp_path / "data" / "XPS-agent",
    )
    project.mkdir()
    (project / ".env.example").write_text("SILICONFLOW_API_KEY=\n")
    source.mkdir()
    for name in ("NF.xlsx", "RO.xlsx"):
        (source / name).write_bytes(b"original" + name.encode())
    reference = source / "reference" / "NF" / "1.pdf"
    reference.parent.mkdir(parents=True)
    reference.write_bytes(b"original pdf")
    for name in ("NF-N", "NF-O", "RO-N", "RO-O"):
        path = project / "final" / name
        path.mkdir(parents=True)
        (path / "spectrum.csv").write_text("energy,intensity\n400,1\n")
    (project / "final" / "old-ml.txt").write_text("DO NOT IMPORT")
    db = StateDB(project / "workspace" / "state" / "xps_agent.sqlite3")
    db.initialize()
    db.upsert_document(
        {
            "doc_key": "original-id",
            "title": "paper",
            "source": "local",
            "local_path": str(reference),
            "status": "evidence_extracted",
        }
    )
    with pytest.raises(ValueError, match="Stop"):
        migrate_storage(project, source, destination, apply=True)
    result = migrate_storage(project, source, destination, apply=True, server_stopped=True)
    assert result["original_data_preserved"]
    assert reference.exists() and (source / "NF.xlsx").exists()
    copied = destination / "raw" / "tables" / "NF.xlsx"
    assert copied.read_bytes() == (source / "NF.xlsx").read_bytes()
    new_db = StateDB(destination / "workspace" / "state" / "xps_agent.sqlite3")
    row = new_db.rows("SELECT * FROM documents")[0]
    assert row["doc_key"] == "original-id" and row["status"] == "evidence_extracted"
    assert row["local_path"] == str(destination / "raw" / "literature" / "NF" / "1.pdf")
    assert not (destination / "raw" / "spectra" / "old-ml.txt").exists()
    for name in (
        "XPS_STORAGE_ROOT",
        "XPS_WORKSPACE_ROOT",
        "XPS_DATA_ROOT",
        "XPS_REFERENCE_ROOT",
        "XPS_LEGACY_ROOT",
    ):
        monkeypatch.delenv(name, raising=False)


def test_migration_never_overwrites_conflicting_data(tmp_path):
    project, source, destination = (
        tmp_path / "project",
        tmp_path / "XPS-914",
        tmp_path / "data" / "XPS-agent",
    )
    project.mkdir()
    source.mkdir()
    for name in ("NF.xlsx", "RO.xlsx"):
        (source / name).write_bytes(b"original")
    (source / "reference").mkdir()
    for name in ("NF-N", "NF-O", "RO-N", "RO-O"):
        (project / "final" / name).mkdir(parents=True)
    target = destination / "raw" / "tables" / "NF.xlsx"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"user-new-data")
    with pytest.raises(ValueError, match="conflict"):
        migrate_storage(project, source, destination, apply=True, server_stopped=True)
    assert target.read_bytes() == b"user-new-data"
