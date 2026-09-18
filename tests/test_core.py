from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from xps_agent.db import StateDB
from xps_agent.features import DescriptorEngine, DescriptorSpec, SpectrumSummary
from xps_agent.utils import normalize_doi


def valid_spec(**overrides):
    payload = {
        "name": "Peak-state balance index",
        "symbol": "peak_state_balance",
        "operation": "normalized_difference",
        "inputs": ["peak_a", "peak_b"],
        "parameters": {"epsilon": 1e-12},
        "equation": "(a-b)/(|a|+|b|)",
        "units": "dimensionless",
        "mechanism_chain": ["peak balance", "local bonding state", "transport"],
        "applicability": ["same element and normalization basis"],
        "confounders": ["peak fit uncertainty"],
        "falsification_tests": ["group permutation", "independent DOI holdout"],
        "expected_direction": "conditional",
        "identifiability": "both inputs are available",
    }
    payload.update(overrides)
    return DescriptorSpec.model_validate(payload)


def test_normalize_doi():
    assert (
        normalize_doi("https://doi.org/10.1038/s41467-024-12345-6.") == "10.1038/s41467-024-12345-6"
    )


def test_known_descriptor_rejected():
    with pytest.raises(ValueError):
        valid_spec(name="HABD")


def test_descriptor_engine():
    frame = pd.DataFrame({"peak_a": [3.0, 1.0], "peak_b": [1.0, 1.0]})
    result = DescriptorEngine.compute_table(frame, valid_spec())
    assert result.iloc[0] == pytest.approx(0.5)
    assert result.iloc[1] == pytest.approx(0.0)


def test_spectrum_summary():
    energy = np.linspace(395, 405, 101)
    intensity = np.exp(-0.5 * ((energy - 400) / 1.2) ** 2)
    result = SpectrumSummary.compute(energy, intensity)
    assert result["centroid_eV"] == pytest.approx(400, abs=0.05)
    assert result["central_80_width_eV"] > 0


def test_database_roundtrip(tmp_path: Path):
    db = StateDB(tmp_path / "state.sqlite3")
    db.initialize()
    db.upsert_document(
        {"doc_key": "doi:10.1/test", "doi": "10.1/test", "title": "Test", "source": "local"}
    )
    rows = db.rows("SELECT title FROM documents")
    assert rows == [{"title": "Test"}]
