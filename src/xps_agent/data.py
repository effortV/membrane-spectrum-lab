from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .config import Settings
from .db import StateDB
from .features import SpectrumSummary, SPECTRUM_SUMMARY_VERSION, is_quality_column
from .utils import json_dumps, sha256_file, utc_now


HEADER_TERMS = (
    "doi",
    "sample",
    "membrane",
    "flux",
    "permeance",
    "rejection",
    "selectivity",
    "pressure",
    "temperature",
    "xps",
    "n1s",
    "o1s",
    "binding",
)

ENERGY_PATTERNS = ("binding energy", "binding_energy", "b.e.", "energy", "ev")
INTENSITY_PATTERNS = ("intensity", "counts", "cps", "signal", "area", "raw", "fit")


def _cell_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def guess_header_row(preview: pd.DataFrame, max_rows: int = 8) -> int:
    best_row = 0
    best_score = float("-inf")
    for row_index in range(min(max_rows, len(preview))):
        values = [_cell_text(value) for value in preview.iloc[row_index].tolist()]
        nonempty = [value for value in values if value]
        if not nonempty:
            continue
        strings = sum(not _looks_numeric(value) for value in nonempty)
        keywords = sum(any(term in value.lower() for term in HEADER_TERMS) for value in nonempty)
        uniqueness = len(set(nonempty)) / max(1, len(nonempty))
        score = strings + 3 * keywords + uniqueness - 0.1 * row_index
        if score > best_score:
            best_row, best_score = row_index, score
    return best_row


def _looks_numeric(value: str) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def flatten_columns(columns: Iterable[Any]) -> list[str]:
    output: list[str] = []
    counts: dict[str, int] = {}
    for index, column in enumerate(columns):
        if isinstance(column, tuple):
            parts = [
                _cell_text(part)
                for part in column
                if _cell_text(part) and not _cell_text(part).startswith("Unnamed")
            ]
            name = " | ".join(parts)
        else:
            name = _cell_text(column)
        name = re.sub(r"\s+", " ", name).strip() or f"unnamed_{index}"
        count = counts.get(name, 0)
        counts[name] = count + 1
        output.append(f"{name}__{count + 1}" if count else name)
    return output


def classify_column(name: str) -> str:
    lower = name.lower()
    if "制备工艺" in name or "制备参数" in name:
        return "preparation"
    if "膜结构参数" in name:
        return "structure"
    if "操作参数" in name or "测试条件" in name:
        return "test_condition"
    if "膜性能参数" in name:
        return "outcome"
    if any(
        term in lower
        for term in (
            "doi",
            "author",
            "reference",
            "paper",
            "sample",
            "membrane",
            "参考文献",
            "序号",
            "膜类型",
            "命名",
        )
    ):
        return "identity"
    if any(
        term in lower
        for term in (
            "flux",
            "permeability",
            "permeation",
            "permeance",
            "rejection",
            "selectivity",
            "mwco",
            "水通量",
            "渗透率",
            "渗透系数",
            "通量",
            "选择性",
            "盐通量",
            "截留",
        )
    ):
        return "outcome"
    if any(term in lower for term in ("xps", "n1s", "o1s", "binding", "peak", "atomic", "峰")):
        return "xps"
    if any(
        term in lower
        for term in (
            "temperature",
            "time",
            "concentration",
            "pressure",
            "ph",
            "monomer",
            "solvent",
            "support",
            "温度",
            "时间",
            "浓度",
            "压力",
            "单体",
            "添加剂",
        )
    ):
        return "condition"
    return "other"


@dataclass
class WorkbookSheet:
    source: Path
    sheet: str
    header_row: int
    frame: pd.DataFrame
    source_rows: list[int]


class DataAuditor:
    def __init__(self, settings: Settings, db: StateDB | None = None):
        self.settings = settings
        self.db = db

    @staticmethod
    def _hierarchical_columns(raw: pd.DataFrame, depth: int = 3) -> list[str]:
        levels = [[_cell_text(value) for value in raw.iloc[row].tolist()] for row in range(depth)]
        top: list[str] = []
        current = ""
        for value in levels[0]:
            if value:
                current = value
            top.append(current)
        second: list[str] = []
        current_second = ""
        previous_top = None
        for top_value, value in zip(top, levels[1], strict=True):
            if top_value != previous_top:
                current_second = ""
                previous_top = top_value
            if value:
                current_second = value
            second.append(current_second)
        names: list[str] = []
        for column_index in range(raw.shape[1]):
            parts: list[str] = []
            for value in (top[column_index], second[column_index], levels[2][column_index]):
                if value and value not in parts and not value.startswith("Unnamed"):
                    parts.append(value)
            names.append(" | ".join(parts) or f"unnamed_{column_index}")
        return flatten_columns(names)

    def read_sheet(self, path: Path, sheet: str) -> WorkbookSheet:
        raw = pd.read_excel(path, sheet_name=sheet, header=None)
        header_numeric = sum(
            _looks_numeric(_cell_text(value))
            for value in raw.head(3).to_numpy().ravel()
            if _cell_text(value)
        )
        header_nonempty = int(raw.head(3).notna().sum().sum())
        if (
            len(raw) >= 4
            and header_nonempty >= raw.shape[1] * 0.5
            and header_numeric / max(1, header_nonempty) < 0.1
            and raw.iloc[:2].isna().any().any()
        ):
            header_row = 2
            frame = raw.iloc[3:].copy()
            frame.columns = self._hierarchical_columns(raw, depth=3)
        else:
            preview = raw.head(10)
            header_row = guess_header_row(preview)
            frame = raw.iloc[header_row + 1 :].copy()
            frame.columns = flatten_columns(raw.iloc[header_row].tolist())
        frame = frame.dropna(axis=0, how="all").dropna(axis=1, how="all")
        source_rows = (frame.index + 1).astype(int).tolist()
        frame = frame.reset_index(drop=True)
        return WorkbookSheet(path, sheet, header_row, frame, source_rows)

    def audit_workbooks(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        sheets: list[dict[str, Any]] = []
        columns: list[dict[str, Any]] = []
        for name in ("NF.xlsx", "RO.xlsx"):
            path = self.settings.data_root / name
            if not path.exists():
                sheets.append({"source": str(path), "error": "missing"})
                continue
            workbook = pd.ExcelFile(path)
            for sheet_name in workbook.sheet_names:
                item = self.read_sheet(path, sheet_name)
                sheets.append(
                    {
                        "source": str(path),
                        "sha256": sha256_file(path),
                        "sheet": sheet_name,
                        "header_row_zero_based": item.header_row,
                        "rows": len(item.frame),
                        "columns": len(item.frame.columns),
                    }
                )
                for column in item.frame.columns:
                    series = item.frame[column]
                    numeric = pd.to_numeric(series, errors="coerce")
                    columns.append(
                        {
                            "source": str(path),
                            "sheet": sheet_name,
                            "column": column,
                            "role": classify_column(column),
                            "non_null": int(series.notna().sum()),
                            "numeric_fraction": float(numeric.notna().mean()),
                            "unique": int(series.nunique(dropna=True)),
                            "unit_warning": self._unit_warning(column),
                        }
                    )
        return sheets, columns

    @staticmethod
    def _unit_warning(column: str) -> str:
        lower = column.lower()
        if "flux" in lower and any(token in lower for token in ("bar-1", "bar⁻¹", "/bar", "mpa-1")):
            return "Header appears pressure-normalized; do not divide by pressure again."
        if "flux" in lower and "pressure" not in lower and "permeance" not in lower:
            return "Confirm whether this is flux or pressure-normalized permeance."
        return ""

    def _read_spectrum_file(self, path: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        if path.suffix.lower() in {".xlsx", ".xls"}:
            sheets = pd.ExcelFile(path).sheet_names
            loaders = [
                (sheet, lambda sheet=sheet: pd.read_excel(path, sheet_name=sheet, header=None))
                for sheet in sheets
            ]
        elif path.suffix.lower() in {".csv", ".txt", ".tsv"}:
            separator = "\t" if path.suffix.lower() in {".txt", ".tsv"} else None
            loaders = [
                ("data", lambda: pd.read_csv(path, sep=separator, engine="python", header=None))
            ]
        else:
            return rows
        for sheet_name, loader in loaders:
            try:
                raw = loader()
                preview = raw.head(10)
                header_row = guess_header_row(preview)
                headers = flatten_columns(raw.iloc[header_row].tolist())
                data = raw.iloc[header_row + 1 :].copy()
                data.columns = headers
                numeric_counts = {
                    column: int(pd.to_numeric(data[column], errors="coerce").notna().sum())
                    for column in data.columns
                }
                energy = [
                    column
                    for column in data.columns
                    if any(term in column.lower() for term in ENERGY_PATTERNS)
                ]
                intensity = [
                    column
                    for column in data.columns
                    if any(term in column.lower() for term in INTENSITY_PATTERNS)
                    and column not in energy
                ]
                rows.append(
                    {
                        "file": str(path),
                        "sheet": sheet_name,
                        "rows": len(data),
                        "columns": len(data.columns),
                        "header_row_zero_based": header_row,
                        "energy_columns": energy,
                        "intensity_columns": intensity,
                        "numeric_columns": [
                            key for key, value in numeric_counts.items() if value >= 3
                        ],
                        "error": "",
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "file": str(path),
                        "sheet": sheet_name,
                        "rows": 0,
                        "columns": 0,
                        "header_row_zero_based": None,
                        "energy_columns": [],
                        "intensity_columns": [],
                        "numeric_columns": [],
                        "error": str(exc),
                    }
                )
        return rows

    def audit_spectra(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for dataset in ("NF", "RO"):
            for element in ("N", "O"):
                root = self.settings.legacy_root / f"{dataset}-{element}"
                if not root.exists():
                    rows.append(
                        {
                            "file": str(root),
                            "dataset": dataset,
                            "element": element,
                            "error": "directory missing",
                        }
                    )
                    continue
                paths = [
                    path
                    for path in root.rglob("*")
                    if path.is_file()
                    and path.suffix.lower() in {".xlsx", ".xls", ".csv", ".txt", ".tsv"}
                ]
                for path in paths:
                    for item in self._read_spectrum_file(path):
                        item.update(
                            {
                                "dataset": dataset,
                                "element": element,
                                "sha256": sha256_file(path),
                            }
                        )
                        rows.append(item)
        return rows

    @staticmethod
    def _spectrum_identity(path: Path) -> tuple[str, str, str, str] | None:
        match = re.match(r"^(NF|RO)-(\d+)-(\d+)-([NO])$", path.stem, flags=re.I)
        if not match:
            return None
        dataset, reference, sample, element = match.groups()
        return dataset.upper(), str(int(reference)), str(int(sample)), element.upper()

    @staticmethod
    def _load_spectrum_xy(path: Path) -> tuple[np.ndarray, np.ndarray]:
        if path.suffix.lower() == ".csv":
            frame = pd.read_csv(path)
        elif path.suffix.lower() in {".xlsx", ".xls"}:
            frame = pd.read_excel(path, sheet_name=0)
        else:
            raise ValueError("Unsupported spectrum file")
        frame.columns = flatten_columns(frame.columns)
        numeric = {
            column: pd.to_numeric(frame[column], errors="coerce") for column in frame.columns
        }
        useful = [column for column, series in numeric.items() if series.notna().sum() >= 5]
        if len(useful) < 2:
            if path.suffix.lower() == ".csv":
                raw = pd.read_csv(path, header=None)
            else:
                raw = pd.read_excel(path, header=None)
            candidates = [
                pd.to_numeric(raw[column], errors="coerce")
                for column in raw.columns
                if pd.to_numeric(raw[column], errors="coerce").notna().sum() >= 5
            ]
            if len(candidates) < 2:
                raise ValueError("Could not identify two numeric spectrum columns")
            energy, intensity = candidates[:2]
        else:
            energy_columns = [
                column
                for column in useful
                if column.lower() in {"x", "be", "binding energy", "binding_energy"}
                or any(term in column.lower() for term in ENERGY_PATTERNS)
            ]
            intensity_columns = [
                column
                for column in useful
                if column.lower() in {"y", "intensity"}
                or any(term in column.lower() for term in INTENSITY_PATTERNS)
            ]
            energy_name = energy_columns[0] if energy_columns else useful[0]
            intensity_choices = [name for name in intensity_columns if name != energy_name]
            if not intensity_choices:
                intensity_choices = [name for name in useful if name != energy_name]
            intensity_name = intensity_choices[0]
            energy, intensity = numeric[energy_name], numeric[intensity_name]
        mask = energy.notna() & intensity.notna() & np.isfinite(energy) & np.isfinite(intensity)
        return energy[mask].to_numpy(float), intensity[mask].to_numpy(float)

    def canonicalize_spectra(self) -> pd.DataFrame:
        records: list[dict[str, Any]] = []
        for dataset in ("NF", "RO"):
            for element in ("N", "O"):
                root = self.settings.legacy_root / f"{dataset}-{element}"
                if not root.exists():
                    continue
                for path in root.rglob("*"):
                    if not path.is_file() or path.suffix.lower() not in {".csv", ".xlsx", ".xls"}:
                        continue
                    identity = self._spectrum_identity(path)
                    if identity is None:
                        continue
                    try:
                        parsed_dataset, reference, sample, parsed_element = identity
                        energy, intensity = self._load_spectrum_xy(path)
                        expected_window = (350, 450) if parsed_element == "N" else (480, 580)
                        if not expected_window[0] <= float(np.median(energy)) <= expected_window[1]:
                            raise ValueError(
                                "Energy coordinates are inconsistent with the labeled N/O 1s eV range; verify units/calibration."
                            )
                        summary = SpectrumSummary.compute(energy, intensity)
                        differences = np.diff(np.sort(np.unique(energy)))
                        quality = {
                            "point_count": int(len(energy)),
                            "energy_min_eV": float(np.min(energy)),
                            "energy_max_eV": float(np.max(energy)),
                            "duplicate_energy_points": int(len(energy) - len(np.unique(energy))),
                            "energy_step_cv": float(
                                np.std(differences) / (np.mean(differences) + 1e-12)
                            )
                            if len(differences)
                            else float("nan"),
                            "zero_intensity_fraction": float(np.mean(np.isclose(intensity, 0))),
                        }
                        record: dict[str, Any] = {
                            "record_id": f"{parsed_dataset}-{reference}-{sample}",
                            "dataset": parsed_dataset,
                            "reference_number": int(reference),
                            "sample_in_reference": int(sample),
                            "element": parsed_element,
                            "spectrum_source_file": str(path),
                            "spectrum_sha256": sha256_file(path),
                            "spectrum_status": "usable",
                            "spectrum_summary_version": SPECTRUM_SUMMARY_VERSION,
                            "spectrum_error": "",
                        }
                        record.update(summary)
                        record.update(quality)
                        records.append(record)
                    except Exception as exc:
                        records.append(
                            {
                                "record_id": path.stem.rsplit("-", 1)[0],
                                "dataset": dataset,
                                "reference_number": None,
                                "sample_in_reference": None,
                                "element": element,
                                "spectrum_source_file": str(path),
                                "spectrum_sha256": sha256_file(path),
                                "spectrum_status": "failed",
                                "spectrum_error": str(exc),
                            }
                        )
        frame = pd.DataFrame(records)
        output = self.settings.workspace_root / "canonical" / "spectrum_features_long.csv"
        frame.to_csv(output, index=False, encoding="utf-8-sig")
        return frame

    @staticmethod
    def _reference_value(value: Any) -> str | None:
        if pd.isna(value):
            return None
        match = re.search(r"\d+", str(value).strip())
        return str(int(match.group(0))) if match else None

    def _doi_reference_map(self) -> dict[tuple[str, str], str]:
        candidates: dict[tuple[str, str], set[str]] = {}
        if self.db is None:
            return {}
        for row in self.db.rows(
            """
            SELECT d.doi, r.dataset, r.reference_number
            FROM document_references r JOIN documents d USING(doc_key)
            WHERE d.doi IS NOT NULL
            """
        ):
            try:
                dataset = row.get("dataset")
                number = row.get("reference_number")
                if dataset and number is not None:
                    key = (str(dataset).upper(), str(int(number)))
                    candidates.setdefault(key, set()).add(str(row["doi"]))
            except (TypeError, ValueError):
                continue
        return {key: next(iter(values)) for key, values in candidates.items() if len(values) == 1}

    def build_model_table(self, spectra: pd.DataFrame | None = None) -> pd.DataFrame:
        source_frames: list[pd.DataFrame] = []
        for dataset in ("NF", "RO"):
            path = self.settings.data_root / f"{dataset}.xlsx"
            if not path.exists():
                continue
            for sheet in pd.ExcelFile(path).sheet_names:
                item = self.read_sheet(path, sheet)
                frame = item.frame.copy()
                reference_columns = [
                    column
                    for column in frame.columns
                    if column.lower() == "ref"
                    or column.lower().endswith(" | ref")
                    or "参考文献" in column
                ]
                if not reference_columns:
                    raise ValueError(f"No reference column found in {path} / {sheet}")
                reference = frame[reference_columns[-1]].map(self._reference_value)
                sample_number = (
                    frame.assign(_ref=reference).groupby("_ref", dropna=False).cumcount() + 1
                )
                record_ids = [
                    f"{dataset}-{ref}-{sample}" if ref else f"{dataset}-unresolved-{row + 1}"
                    for row, (ref, sample) in enumerate(zip(reference, sample_number, strict=True))
                ]
                frame.insert(0, "record_id", record_ids)
                frame.insert(1, "dataset", dataset)
                frame.insert(2, "reference_number", reference)
                frame.insert(3, "sample_in_reference", sample_number)
                frame.insert(4, "_source_workbook", str(path))
                frame.insert(5, "_source_sheet", sheet)
                frame.insert(6, "_source_row", item.source_rows)
                source_frames.append(frame)
        model = (
            pd.concat(source_frames, ignore_index=True, sort=False)
            if source_frames
            else pd.DataFrame()
        )
        if spectra is None:
            spectrum_path = (
                self.settings.workspace_root / "canonical" / "spectrum_features_long.csv"
            )
            spectra = pd.read_csv(spectrum_path) if spectrum_path.exists() else pd.DataFrame()
        if not spectra.empty:
            usable = spectra[spectra["spectrum_status"] == "usable"].copy()
            value_columns = [
                column
                for column in usable.columns
                if column
                not in {
                    "record_id",
                    "dataset",
                    "reference_number",
                    "sample_in_reference",
                    "element",
                    "spectrum_status",
                    "spectrum_error",
                }
            ]
            for element, group_frame in usable.groupby("element"):
                # Two different files for one record/element are not interchangeable fits.
                conflicts = group_frame.groupby("record_id")["spectrum_sha256"].nunique()
                group_frame = group_frame[
                    ~group_frame["record_id"].isin(conflicts[conflicts > 1].index)
                ]
                part = (
                    group_frame[["record_id", *value_columns]].drop_duplicates("record_id").copy()
                )
                part = part.rename(
                    columns={column: f"{element}__{column}" for column in value_columns}
                )
                model = model.merge(part, on="record_id", how="left", validate="one_to_one")
                model[f"{element}__linkage_basis"] = np.where(
                    model[f"{element}__spectrum_source_file"].notna(),
                    "within_reference_row_order_unverified",
                    "unlinked",
                )
        doi_map = self._doi_reference_map()
        if not model.empty:
            model["doi_group"] = [
                doi_map.get((str(dataset), str(reference)), "") if pd.notna(reference) else ""
                for dataset, reference in zip(
                    model["dataset"], model["reference_number"], strict=True
                )
            ]
            model["validation_group"] = [
                doi if doi else f"{dataset}:ref:{reference}"
                for doi, dataset, reference in zip(
                    model["doi_group"], model["dataset"], model["reference_number"], strict=True
                )
            ]
        destination = self.settings.workspace_root / "canonical" / "model_table.csv"
        model.to_csv(destination, index=False, encoding="utf-8-sig")
        return model

    def run(self) -> dict[str, Any]:
        output = self.settings.workspace_root / "canonical"
        output.mkdir(parents=True, exist_ok=True)
        sheets, columns = self.audit_workbooks()
        spectra = self.audit_spectra()
        pd.DataFrame(sheets).to_csv(
            output / "workbook_inventory.csv", index=False, encoding="utf-8-sig"
        )
        pd.DataFrame(spectra).to_csv(
            output / "spectra_inventory.csv", index=False, encoding="utf-8-sig"
        )
        spectrum_features = self.canonicalize_spectra()
        model_table = self.build_model_table(spectrum_features)
        for column in [name for name in model_table.columns if name.startswith(("N__", "O__"))]:
            numeric = pd.to_numeric(model_table[column], errors="coerce")
            if numeric.notna().mean() < 0.1:
                continue
            columns.append(
                {
                    "source": "canonical spectrum summary",
                    "sheet": "model_table",
                    "column": column,
                    "role": "spectrum_quality" if is_quality_column(column) else "spectrum_shape",
                    "non_null": int(model_table[column].notna().sum()),
                    "numeric_fraction": float(numeric.notna().mean()),
                    "unique": int(model_table[column].nunique(dropna=True)),
                    "unit_warning": "Within-spectrum shape only; not cross-element stoichiometry.",
                }
            )
        pd.DataFrame(columns).to_csv(
            output / "column_inventory.csv", index=False, encoding="utf-8-sig"
        )
        usable_spectra = 0
        paired = 0
        if not spectrum_features.empty:
            usable = spectrum_features[spectrum_features["spectrum_status"] == "usable"]
            usable_spectra = len(usable)
            paired = int((usable.groupby("record_id")["element"].nunique() >= 2).sum())
        profile = {
            "spectrum_summary_version": SPECTRUM_SUMMARY_VERSION,
            "created_at": utc_now(),
            "policy": "NF.xlsx and RO.xlsx are authoritative; final is read-only spectra provenance.",
            "workbook_sheets": sheets,
            "column_count": len(columns),
            "spectrum_sheet_count": len(spectra),
            "spectrum_errors": sum(bool(item.get("error")) for item in spectra),
            "canonical_spectra_usable": usable_spectra,
            "paired_n_o_records": paired,
            "model_table_rows": len(model_table),
            "model_table_sha256": sha256_file(output / "model_table.csv"),
            "spectrum_linkage_policy": "Row-order linkage is provisional; verify sample IDs against each paper before physical claims.",
            "doi_group_rows_resolved": int(
                (
                    model_table.get("doi_group", pd.Series(dtype=str)).fillna("").astype(str) != ""
                ).sum()
            ),
            "warnings": [
                "Do not infer cross-element stoichiometry from separately normalized N/O spectra.",
                "Verify flux versus permeance units before any pressure normalization.",
                "Keep DOI/article groups intact across NF and RO during validation.",
                "Retain spectrum extraction and fit version hashes; propagate fit uncertainty.",
                "Digitized sampling density is not a physical observable; shape integrals use energy-grid weights.",
                "Spectrum/sample association based on within-paper row order is unverified.",
                "Unresolved DOI fallback groups cannot establish independence across NF and RO.",
            ],
        }
        (output / "data_profile.json").write_text(json_dumps(profile), encoding="utf-8")
        return profile

    def export_authoritative_tables(self) -> list[Path]:
        output = self.settings.workspace_root / "canonical" / "source_tables"
        output.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        for workbook_name in ("NF.xlsx", "RO.xlsx"):
            path = self.settings.data_root / workbook_name
            if not path.exists():
                continue
            for sheet in pd.ExcelFile(path).sheet_names:
                item = self.read_sheet(path, sheet)
                destination = output / f"{path.stem}__{re.sub(r'[^A-Za-z0-9_-]+', '_', sheet)}.csv"
                frame = item.frame.copy()
                frame.insert(0, "_source_row", item.source_rows)
                frame.insert(0, "_source_sheet", sheet)
                frame.insert(0, "_source_workbook", str(path))
                frame.to_csv(destination, index=False, encoding="utf-8-sig")
                written.append(destination)
        return written
