from __future__ import annotations

import re
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.stats import skew


KNOWN_DESCRIPTOR_NAMES = {
    "habd",
    "hcd",
    "had",
    "dnc",
    "cross-linking degree",
    "crosslinking degree",
    "o2/o1",
    "o2_o1",
}
SPECTRUM_SUMMARY_VERSION = "2.0-grid-invariant"
QUALITY_FIELDS = {
    "point_count",
    "energy_min_eV",
    "energy_max_eV",
    "duplicate_energy_points",
    "energy_step_cv",
    "zero_intensity_fraction",
    "integrated_area_arb_eV",
}


def is_known_descriptor(value: str) -> bool:
    compact = value.strip().lower()
    words = set(re.split(r"[^a-z0-9]+", compact))
    alphanumeric = "".join(character for character in compact if character.isalnum())
    return (
        compact in KNOWN_DESCRIPTOR_NAMES
        or bool(words & {"habd", "hcd", "had", "dnc"})
        or any(token in alphanumeric for token in ("o2o1", "o1o2", "crosslinkingdegree"))
        or "交联度" in compact
    )


def is_quality_column(value: str) -> bool:
    return value.split("__", 1)[-1] in QUALITY_FIELDS


class DescriptorSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=2, max_length=160)
    symbol: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_]{1,63}$")
    operation: Literal[
        "ratio",
        "log_ratio",
        "difference",
        "normalized_difference",
        "product",
        "weighted_sum",
    ]
    inputs: list[str] = Field(min_length=2, max_length=8)
    parameters: dict[str, float] = Field(default_factory=dict)
    equation: str
    units: str
    mechanism_chain: list[str] = Field(min_length=3)
    applicability: list[str] = Field(min_length=1)
    confounders: list[str] = Field(min_length=1)
    falsification_tests: list[str] = Field(min_length=2)
    expected_direction: str
    evidence_ids: list[str] = Field(default_factory=list)
    novelty_queries: list[str] = Field(default_factory=list)
    identifiability: str
    dataset_scope: list[Literal["NF", "RO"]] = Field(default_factory=list)
    input_bounds: dict[str, tuple[float | None, float | None]] = Field(default_factory=dict)
    status: Literal["proposed"] = "proposed"

    @field_validator("name", "symbol")
    @classmethod
    def not_known_descriptor(cls, value: str) -> str:
        if is_known_descriptor(value):
            raise ValueError("Known descriptors are baselines only, not new hypotheses")
        return value.strip()

    @model_validator(mode="after")
    def validate_operation(self) -> "DescriptorSpec":
        if len(set(self.inputs)) != len(self.inputs):
            raise ValueError("Descriptor inputs must be unique")
        pair_operations = {"ratio", "log_ratio", "difference", "normalized_difference"}
        if self.operation in pair_operations and len(self.inputs) != 2:
            raise ValueError("This operation requires exactly two inputs")
        if any(not np.isfinite(value) for value in self.parameters.values()):
            raise ValueError("Descriptor parameters must be finite")
        if self.parameters.get("epsilon", 1e-12) <= 0:
            raise ValueError("epsilon must be positive")
        allowed = {"epsilon"}
        if self.operation == "weighted_sum":
            allowed.update(f"w{index + 1}" for index in range(len(self.inputs)))
        if set(self.parameters) - allowed:
            raise ValueError("Unknown operation parameters")
        for name, (lower, upper) in self.input_bounds.items():
            if name not in self.inputs:
                raise ValueError("Bounds may only refer to descriptor inputs")
            if any(value is not None and not np.isfinite(value) for value in (lower, upper)):
                raise ValueError("Bounds must be finite")
            if lower is not None and upper is not None and lower > upper:
                raise ValueError("Lower bound cannot exceed upper bound")
        return self


class DescriptorEngine:
    EPS = 1e-12

    @classmethod
    def compute_table(cls, frame: pd.DataFrame, spec: DescriptorSpec) -> pd.Series:
        missing = [column for column in spec.inputs if column not in frame.columns]
        if missing:
            raise KeyError(f"Descriptor inputs absent from table: {missing}")
        values = [
            pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
            for column in spec.inputs
        ]
        a, b = values[0], values[1]
        epsilon = float(spec.parameters.get("epsilon", cls.EPS))
        if spec.operation == "ratio":
            output = a / b.where(b.abs() > epsilon)
        elif spec.operation == "log_ratio":
            output = np.log(a.where(a > epsilon) / b.where(b > epsilon))
        elif spec.operation == "difference":
            output = a - b
        elif spec.operation == "normalized_difference":
            denominator = a.abs() + b.abs()
            output = (a - b) / denominator.where(denominator > epsilon)
        elif spec.operation == "product":
            output = a * b
            for value in values[2:]:
                output = output * value
        elif spec.operation == "weighted_sum":
            weights = [
                float(spec.parameters.get(f"w{index + 1}", 1.0)) for index in range(len(values))
            ]
            output = sum(weight * value for weight, value in zip(weights, values, strict=True))
        else:  # pragma: no cover - Pydantic blocks unknown operations
            raise ValueError(f"Unsupported operation: {spec.operation}")
        eligible = pd.Series(True, index=frame.index)
        if spec.dataset_scope:
            if "dataset" not in frame:
                raise KeyError("dataset is required by the descriptor applicability gate")
            eligible &= frame["dataset"].isin(spec.dataset_scope)
        for column, (lower, upper) in spec.input_bounds.items():
            value = pd.to_numeric(frame[column], errors="coerce")
            if lower is not None:
                eligible &= value >= lower
            if upper is not None:
                eligible &= value <= upper
        return (
            pd.Series(output, index=frame.index, name=spec.symbol)
            .where(eligible)
            .replace([np.inf, -np.inf], np.nan)
        )


class SpectrumSummary:
    """Numerically grounded descriptors for a single spectrum, without chemical assignment."""

    @staticmethod
    def compute(energy: np.ndarray, intensity: np.ndarray) -> dict[str, float]:
        x = np.asarray(energy, dtype=float)
        y = np.asarray(intensity, dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        if x.size < 5:
            raise ValueError("At least five finite spectrum points are required")
        order = np.argsort(x)
        x, y = x[order], y[order]
        unique_x, inverse = np.unique(x, return_inverse=True)
        y = np.bincount(inverse, weights=y) / np.bincount(inverse)
        x = unique_x
        if len(x) < 5 or x[-1] <= x[0]:
            raise ValueError("At least five distinct energy coordinates are required")
        y = y - np.nanmin(y)
        area = float(trapezoid(y, x))
        if area <= 0:
            raise ValueError("Spectrum has no positive integrated signal after baseline shift")
        density = y / area
        centroid = float(trapezoid(x * density, x))
        variance = float(trapezoid(((x - centroid) ** 2) * density, x))
        uniform_x = np.linspace(x[0], x[-1], 256)
        uniform_y = np.interp(uniform_x, x, y)
        probability = uniform_y / np.sum(uniform_y)
        entropy = float(-np.sum(probability * np.log(probability + 1e-12)))
        cumulative = cumulative_trapezoid(density, x, initial=0)
        q10, q50, q90 = [float(np.interp(q, cumulative, x)) for q in (0.1, 0.5, 0.9)]
        positive = density > 0
        entropy_integrand = np.zeros_like(density)
        entropy_integrand[positive] = density[positive] * np.log(density[positive] * (x[-1] - x[0]))
        differential_entropy = -float(trapezoid(entropy_integrand, x))
        high_x = np.concatenate(([centroid], x[x > centroid]))
        high_y = np.interp(high_x, x, density)
        return {
            "integrated_area_arb_eV": area,
            "centroid_eV": centroid,
            "spread_eV": float(np.sqrt(max(0.0, variance))),
            "spectral_entropy": entropy,
            "spectral_entropy_effective_fraction": float(np.exp(min(0.0, differential_entropy))),
            "intensity_skewness": float(skew(uniform_y, bias=False, nan_policy="omit")),
            "energy_skewness": float(trapezoid((x - centroid) ** 3 * density, x) / variance**1.5)
            if variance > 0
            else 0.0,
            "energy_q10_eV": q10,
            "energy_q50_eV": q50,
            "energy_q90_eV": q90,
            "central_80_width_eV": q90 - q10,
            "high_energy_fraction_above_centroid": float(trapezoid(high_y, high_x)),
        }
