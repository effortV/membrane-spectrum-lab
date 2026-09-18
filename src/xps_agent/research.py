"""Preparation/XPS/structure/performance research with grouped validation.

Associations are predictive tests, not identified causal effects or new physics.
Original measurements and units are never overwritten by this module.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable
import uuid

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import Settings
from .db import StateDB
from .features import is_quality_column, SPECTRUM_SUMMARY_VERSION
from .ml import _metrics
from .utils import json_dumps, sha256_file, sha256_text, utc_now


ROLES = [
    "preparation",
    "xps",
    "structure",
    "structure_proxy",
    "test_condition",
    "performance",
    "identity",
    "quality",
    "unassigned",
]
ROLE_LABELS = {
    "preparation": "制备",
    "xps": "XPS",
    "structure": "独立结构表征",
    "structure_proxy": "XPS 推导结构代理量",
    "test_condition": "性能测试条件",
    "performance": "膜性能",
    "identity": "编号/来源",
    "quality": "提取质量",
    "unassigned": "待分类",
}
TASKS = {
    "preparation_to_xps": {"label": "制备 → XPS", "inputs": ["preparation"], "target": "xps"},
    "preparation_xps_to_structure": {
        "label": "制备 + XPS → 结构",
        "inputs": ["preparation", "xps"],
        "target": "structure",
    },
    "xps_to_structure": {"label": "XPS → 结构", "inputs": ["xps"], "target": "structure"},
    "xps_to_performance": {"label": "XPS → 性能", "inputs": ["xps"], "target": "performance"},
    "structure_to_performance": {
        "label": "结构 → 性能",
        "inputs": ["structure"],
        "target": "performance",
    },
    "chain_to_performance": {
        "label": "制备 + XPS + 结构 → 性能",
        "inputs": ["preparation", "xps", "structure"],
        "target": "performance",
    },
}


def scientific_role(column: str) -> str:
    lower = column.lower()
    if column.startswith("_") or lower in {
        "record_id",
        "dataset",
        "sheet",
        "source_row",
        "reference_number",
        "doi_group",
        "validation_group",
        "source_excel_row",
        "source_file",
        "sequence_number",
    }:
        return "identity"
    if any(
        word in lower
        for word in (
            "source",
            "sha256",
            "linkage",
            "参考文献",
            "序号",
            "命名",
            "膜类型",
            "doi",
            "图片",
            "图像",
        )
    ):
        return "identity"
    if is_quality_column(column):
        return "quality"
    # Hierarchical source sections take precedence: support MWCO is preparation,
    # not the selective-layer target; membrane R(NaCl) is performance.
    if "制备工艺" in column or "制备参数" in column:
        return "preparation"
    if "操作参数" in column or "测试条件" in column:
        return "test_condition"
    if "膜性能参数" in column:
        return "performance"
    if any(word in lower for word in ("habd", "hcd", "had计算", "dnc", "交联度", "crosslink")):
        return "structure_proxy"
    if "膜结构参数" in column:
        return "structure"
    if lower.startswith(("n__", "o__", "c__")) or any(
        word in lower for word in ("xps", "binding_energy", "spectral_entropy")
    ):
        return "xps"
    if any(
        word in lower
        for word in (
            "thickness",
            "roughness",
            "contact_angle",
            "zeta",
            "pore",
            "厚度",
            "粗糙",
            "接触角",
            "孔径",
            "孔半径",
        )
    ):
        return "structure"
    if any(
        word in lower
        for word in ("flux", "permeance", "rejection", "selectivity", "水通量", "截留")
    ):
        return "performance"
    if any(word in lower for word in ("pressure", "feed", "test_ph", "压力")):
        return "test_condition"
    if any(
        word in lower
        for word in (
            "monomer",
            "ip_time",
            "ip_temperature",
            "support",
            "单体",
            "反应时间",
            "后处理",
        )
    ):
        return "preparation"
    return "unassigned"


def clean_series(series: pd.Series) -> pd.Series:
    values = series.astype(object).copy()
    markers = (
        values.astype(str)
        .str.strip()
        .str.lower()
        .isin({"", "-", "—", "/", "nan", "none", "n/a", "na", "not reported"})
    )
    values.loc[markers | values.isna()] = np.nan
    return values


def field_catalog(
    frame: pd.DataFrame,
    overrides: dict[str, str] | None = None,
    candidate_symbols: list[str] | None = None,
) -> pd.DataFrame:
    overrides = overrides or {}
    rows = []
    for column in frame.columns:
        clean = clean_series(frame[column])
        numeric = pd.to_numeric(clean, errors="coerce").replace([np.inf, -np.inf], np.nan)
        role = scientific_role(str(column))
        if role == "unassigned" and column in (candidate_symbols or []):
            role = "xps"
        # Provenance/quality columns cannot be relabelled as scientific predictors.
        if role not in {"identity", "quality"} and overrides.get(column) in ROLES:
            role = overrides[column]
        fraction = numeric.notna().sum() / max(1, clean.notna().sum())
        rows.append(
            {
                "column": str(column),
                "scientific_role": role,
                "type": "numeric" if fraction >= 0.8 else "categorical",
                "non_null": int(clean.notna().sum()),
                "numeric_count": int(numeric.notna().sum()),
                "coverage": float(clean.notna().mean()),
                "definition": ROLE_LABELS[role],
                "origin": "agent_candidate_unvalidated"
                if column in (candidate_symbols or [])
                else "original_table_or_spectrum_summary",
            }
        )
    return pd.DataFrame(rows)


def load_role_overrides(settings: Settings) -> dict[str, str]:
    path = settings.catalog_root / "scientific_roles.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {key: value for key, value in payload.get("roles", {}).items() if value in ROLES}


def save_role_overrides(settings: Settings, roles: dict[str, str]) -> None:
    if any(value not in ROLES for value in roles.values()):
        raise ValueError("Unknown scientific field group")
    settings.catalog_root.mkdir(parents=True, exist_ok=True)
    path = settings.catalog_root / "scientific_roles.json"
    temp = path.with_suffix(".tmp")
    temp.write_text(json_dumps({"updated_at": utc_now(), "roles": roles}), encoding="utf-8")
    temp.replace(path)


def current_research_table(settings: Settings) -> Path:
    base = settings.workspace_root / "canonical" / "model_table.csv"
    enriched = base.with_name("model_table_enriched.csv")
    if enriched.exists():
        try:
            manifest = json.loads(
                enriched.with_suffix(".manifest.json").read_text(encoding="utf-8")
            )
            if (
                manifest.get("source_sha256") == sha256_file(base)
                and manifest.get("destination_sha256") == sha256_file(enriched)
                and manifest.get("spectrum_summary_version") == SPECTRUM_SUMMARY_VERSION
            ):
                return enriched
        except (OSError, ValueError):
            pass
    return base


def prepare_research_data(
    frame: pd.DataFrame,
    catalog: pd.DataFrame,
    task: str,
    target: str,
    groups: dict[str, list[str]],
    controls: list[str],
    *,
    datasets: list[str],
    allow_unresolved: bool = False,
    measured_intermediates: bool = True,
) -> tuple[pd.DataFrame, dict[str, list[str]], dict[str, Any]]:
    if task not in TASKS:
        raise ValueError("Unknown research relationship")
    spec = TASKS[task]
    role_map = dict(zip(catalog["column"], catalog["scientific_role"], strict=True))
    type_map = dict(zip(catalog["column"], catalog["type"], strict=True))
    if role_map.get(target) != spec["target"]:
        raise ValueError(
            "The target must belong to the selected research relationship; XPS-derived structure proxies are not independent structure targets"
        )
    if not datasets or "dataset" not in frame or "doi_group" not in frame:
        raise ValueError("Dataset selection and DOI grouping are required")
    if not groups or any(key not in spec["inputs"] for key in groups):
        raise ValueError("Predictor groups do not match the selected relationship")
    for key in spec["inputs"]:
        if not groups.get(key):
            raise ValueError(f"Select at least one {ROLE_LABELS[key]} variable")
        if any(role_map.get(name) != key for name in groups[key]):
            raise ValueError(f"Variables in {key} have incorrect scientific roles")
    if any(role_map.get(name) != "test_condition" for name in controls):
        raise ValueError("Controls must be performance-test conditions, not outcomes")
    features = [name for names in groups.values() for name in names] + controls
    if len(features) != len(set(features)) or target in features or len(features) > 40:
        raise ValueError("Predictors must be unique, exclude the target, and total at most 40")
    required = [target, *features]
    if any(name not in frame for name in required):
        raise ValueError("Selected columns do not exist in the table")
    data = frame[list(dict.fromkeys([*required, "dataset", "doi_group"]))].copy()
    data["_input_row_index"] = np.arange(len(frame))
    data["_input_record_id"] = (
        frame["record_id"].astype(str) if "record_id" in frame else frame.index.astype(str)
    )
    data[target] = pd.to_numeric(clean_series(data[target]), errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    for name in features:
        clean = clean_series(data[name])
        data[name] = (
            pd.to_numeric(clean, errors="coerce").replace([np.inf, -np.inf], np.nan)
            if type_map[name] == "numeric"
            else clean.map(lambda value: str(value).strip() if pd.notna(value) else np.nan)
        )
    selected = data["dataset"].astype(str).isin(datasets)
    has_target = data[target].notna()
    doi = data["doi_group"].fillna("").astype(str).str.strip()
    has_doi = ~doi.str.lower().isin({"", "none", "nan", "nat", "inf", "-inf"})
    # A PNAS supplementary-page URL is not an additional independent article.
    # Preserve the original DOI column; canonicalize this explicitly recognized URL tail only.
    data["_group"] = doi.str.lower().map(
        lambda value: re.sub(r"^(10\.1073/pnas\.[a-z0-9.]+?)/-/dcsupplemental.*$", r"\1", value)
    )
    if allow_unresolved:
        if "validation_group" not in frame:
            raise ValueError("Missing fallback validation groups")
        fallback = frame["validation_group"].fillna("").astype(str).str.strip()
        fallback_valid = ~fallback.str.lower().isin({"", "none", "nan", "nat", "inf", "-inf"})
        data.loc[~has_doi, "_group"] = fallback[~has_doi]
        has_group = has_doi | fallback_valid
    else:
        has_group = has_doi
    eligible = selected & has_target & has_group
    intermediates = [name for key in ("xps", "structure") for name in groups.get(key, [])]
    before_intermediates = int(eligible.sum())
    if measured_intermediates and intermediates:
        eligible &= data[intermediates].notna().all(axis=1)
    data = data.loc[eligible].reset_index(drop=True)
    sets: dict[str, list[str]] = (
        {"test_conditions": controls} if controls else {"training_mean": []}
    )
    for key in spec["inputs"]:
        sets[key] = list(dict.fromkeys([*controls, *groups[key]]))
    combined = list(dict.fromkeys(features))
    if len(groups) > 1:
        sets["combined"] = combined
    origin = dict(
        zip(
            catalog["column"],
            catalog.get("origin", pd.Series("original", index=catalog.index)),
            strict=True,
        )
    )
    candidates = [
        name for name in groups.get("xps", []) if origin.get(name) == "agent_candidate_unvalidated"
    ]
    observables = [name for name in groups.get("xps", []) if name not in candidates]
    if candidates and observables:
        sets["xps_observables"] = list(dict.fromkeys([*controls, *observables]))
        sets["xps_candidate_descriptors"] = list(dict.fromkeys([*controls, *candidates]))
    report = {
        "input_rows": len(frame),
        "dataset_rows": int(selected.sum()),
        "rows_scored": len(data),
        "groups": int(data["_group"].nunique()),
        "missing_target": int((selected & ~has_target).sum()),
        "missing_doi_or_group": int((selected & has_target & ~has_group).sum()),
        "missing_measured_intermediates": before_intermediates - len(data),
        "measured_intermediates": measured_intermediates,
        "allow_unresolved": allow_unresolved,
        "datasets": datasets,
        "preparation_categories_supported": True,
    }
    return data, sets, report


class RelationshipEvaluator:
    def __init__(
        self,
        settings: Settings,
        db: StateDB,
        *,
        progress: Callable[..., None] | None = None,
        check_cancelled: Callable[[], None] | None = None,
    ):
        self.settings, self.db = settings, db
        self.progress = progress or (lambda **_: None)
        self.check = check_cancelled or (lambda: None)

    @staticmethod
    def candidates(model: str, seed: int):
        if model == "ridge":
            return [Ridge(alpha=alpha) for alpha in (1.0, 10.0)]
        cls = {"random_forest": RandomForestRegressor, "extra_trees": ExtraTreesRegressor}.get(
            model
        )
        if cls is None:
            raise ValueError("Supported models: ridge/random_forest/extra_trees")
        return [
            cls(n_estimators=120, max_depth=depth, min_samples_leaf=3, random_state=seed, n_jobs=2)
            for depth in (5, None)
        ]

    @staticmethod
    def pipeline(data: pd.DataFrame, names: list[str], model) -> Pipeline:
        numeric = [name for name in names if pd.api.types.is_numeric_dtype(data[name])]
        categorical = [name for name in names if name not in numeric]
        transforms = []
        if numeric:
            transforms.append(
                (
                    "numeric",
                    Pipeline(
                        [
                            (
                                "impute",
                                SimpleImputer(
                                    strategy="median", keep_empty_features=True, add_indicator=True
                                ),
                            ),
                            ("scale", StandardScaler()),
                        ]
                    ),
                    numeric,
                )
            )
        if categorical:
            transforms.append(
                (
                    "categorical",
                    Pipeline(
                        [
                            (
                                "impute",
                                SimpleImputer(
                                    strategy="constant",
                                    fill_value="missing",
                                    keep_empty_features=True,
                                ),
                            ),
                            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                        ]
                    ),
                    categorical,
                )
            )
        return Pipeline([("prepare", ColumnTransformer(transforms)), ("model", model)])

    def evaluate(
        self,
        table: Path,
        task: str,
        target: str,
        groups: dict[str, list[str]],
        controls: list[str],
        *,
        datasets: list[str],
        outer_splits: int = 3,
        inner_splits: int = 2,
        models: list[str] | None = None,
        allow_unresolved: bool = False,
        measured_intermediates: bool = True,
        seed: int = 20260917,
    ) -> dict[str, Any]:
        self.check()
        if not 3 <= outer_splits <= 10 or not 2 <= inner_splits <= 8:
            raise ValueError("Invalid grouped fold counts")
        models = list(dict.fromkeys(models or ["ridge", "random_forest"]))
        for model in models:
            self.candidates(model, seed)
        if (
            table.name == "model_table_enriched.csv"
            and current_research_table(self.settings).resolve() != table.resolve()
        ):
            raise ValueError("Candidate table is stale; recalculate descriptors first")
        digest = sha256_file(table)
        frame = pd.read_csv(table, low_memory=False)
        symbols = [
            json.loads(row["spec_json"]).get("symbol")
            for row in self.db.rows("SELECT spec_json FROM hypotheses WHERE status='calculable'")
        ]
        catalog = field_catalog(frame, load_role_overrides(self.settings), symbols)
        data, sets, cohort = prepare_research_data(
            frame,
            catalog,
            task,
            target,
            groups,
            controls,
            datasets=datasets,
            allow_unresolved=allow_unresolved,
            measured_intermediates=measured_intermediates,
        )
        if len(data) < 12 or cohort["groups"] < 3:
            raise ValueError(
                f"有效样本 {len(data)}、独立文献组 {cohort['groups']}；至少需要 12 条样本和 3 组，请减少稀疏特征或补齐数据。"
            )
        folds = list(
            GroupKFold(n_splits=min(outer_splits, cohort["groups"])).split(
                data, data[target], data["_group"]
            )
        )
        fits_total = sum(
            sum(
                (
                    len(self.candidates(model, seed))
                    * min(inner_splits, data.iloc[train]["_group"].nunique())
                    + 1
                )
                for model in models
            )
            * sum(bool(names) for names in sets.values())
            for train, _ in folds
        )
        self.progress(
            stage="准备分组关系研究", fits_total=fits_total, fits_finished=0, outer_total=len(folds)
        )
        finished = 0
        predictions, selections, split_manifest, metrics = [], [], [], []
        for fold, (train, test) in enumerate(folds, start=1):
            self.check()
            train_data, test_data = data.iloc[train], data.iloc[test]
            if set(train_data["_group"]) & set(test_data["_group"]):
                raise RuntimeError("DOI group leakage")
            inner = list(
                GroupKFold(n_splits=min(inner_splits, train_data["_group"].nunique())).split(
                    train_data, train_data[target], train_data["_group"]
                )
            )
            split_manifest.append(
                {
                    "fold": fold,
                    "train_groups": sorted(set(train_data["_group"])),
                    "test_groups": sorted(set(test_data["_group"])),
                    "train_input_rows": train_data["_input_row_index"].tolist(),
                    "test_input_rows": test_data["_input_row_index"].tolist(),
                }
            )
            for feature_set, names in sets.items():
                families = models if names else ["training_mean"]
                for family in families:
                    self.check()
                    self.progress(
                        stage="外层分组模型比较",
                        outer_fold=fold,
                        feature_set=feature_set,
                        model=family,
                    )
                    if not names:
                        predicted = np.full(len(test), train_data[target].mean())
                    else:
                        empty = [name for name in names if train_data[name].isna().all()]
                        if empty:
                            raise ValueError(
                                f"第 {fold} 折的训练文献完全缺少 {empty}，不能可靠填补。"
                            )
                        best, best_mae = None, float("inf")
                        for candidate_index, candidate in enumerate(
                            self.candidates(family, seed + fold), start=1
                        ):
                            errors = []
                            for inner_fold, (inside, validation) in enumerate(inner, start=1):
                                self.check()
                                self.progress(
                                    stage="内层训练和验证",
                                    outer_fold=fold,
                                    inner_fold=inner_fold,
                                    candidate_index=candidate_index,
                                    fits_finished=finished,
                                )
                                pipe = self.pipeline(train_data.iloc[inside], names, candidate)
                                pipe.fit(
                                    train_data.iloc[inside][names], train_data.iloc[inside][target]
                                )
                                estimate = pipe.predict(train_data.iloc[validation][names])
                                # Give each validation paper equal weight during selection.
                                residuals = pd.DataFrame(
                                    {
                                        "group": train_data.iloc[validation]["_group"].to_numpy(),
                                        "absolute_error": np.abs(
                                            train_data.iloc[validation][target].to_numpy()
                                            - estimate
                                        ),
                                    }
                                )
                                errors.append(
                                    float(
                                        residuals.groupby("group")["absolute_error"].mean().mean()
                                    )
                                )
                                finished += 1
                                self.progress(fits_finished=finished)
                            if np.mean(errors) < best_mae:
                                best, best_mae = candidate, float(np.mean(errors))
                        self.check()
                        pipe = self.pipeline(train_data, names, best)
                        pipe.fit(train_data[names], train_data[target])
                        predicted = pipe.predict(test_data[names])
                        finished += 1
                        self.progress(stage="已完成一个外层模型", fits_finished=finished)
                        selections.append(
                            {
                                "fold": fold,
                                "feature_set": feature_set,
                                "model": family,
                                "inner_group_balanced_mae": best_mae,
                                "parameters": {
                                    key: value
                                    for key, value in best.get_params().items()
                                    if isinstance(value, (int, float, str, type(None), bool))
                                },
                            }
                        )
                    metrics.append(
                        {
                            "fold": fold,
                            "feature_set": feature_set,
                            "model": family,
                            **_metrics(test_data[target].to_numpy(), predicted),
                        }
                    )
                    for (_, row), estimate in zip(test_data.iterrows(), predicted, strict=True):
                        predictions.append(
                            {
                                "fold": fold,
                                "feature_set": feature_set,
                                "model": family,
                                "record_id": row["_input_record_id"],
                                "input_row_index": int(row["_input_row_index"]),
                                "doi_group": row["_group"],
                                "observed": float(row[target]),
                                "predicted": float(estimate),
                            }
                        )
        self.check()
        if sha256_file(table) != digest:
            raise ValueError("Research table changed during training; no result committed")
        prediction_frame = pd.DataFrame(predictions)
        summaries = []
        for (feature_set, model), part in prediction_frame.groupby(["feature_set", "model"]):
            per_group = (
                part.assign(error=(part["observed"] - part["predicted"]).abs())
                .groupby("doi_group")["error"]
                .mean()
            )
            summaries.append(
                {
                    "feature_set": feature_set,
                    "model": model,
                    "group_balanced_mae": float(per_group.mean()),
                    **_metrics(part["observed"].to_numpy(), part["predicted"].to_numpy()),
                }
            )
        run_id = uuid.uuid4().hex
        run_dir = self.settings.workspace_root / "runs" / f"relationship_{run_id}"
        run_dir.mkdir(parents=True, exist_ok=False)
        result = {
            "experiment_id": run_id,
            "created_at": utc_now(),
            "research_task": task,
            "relationship": TASKS[task]["label"],
            "target": target,
            "inputs": groups,
            "controls": controls,
            "feature_sets": sets,
            "cohort": cohort,
            "out_of_fold": summaries,
            "data_hash": digest,
            "table": str(table),
            "split_hash": sha256_text(json_dumps(split_manifest)),
            "code_sha256": sha256_file(Path(__file__)),
            "seed": seed,
            "field_roles": catalog[
                catalog["column"].isin(
                    [target, *[name for names in sets.values() for name in names]]
                )
            ].to_dict(orient="records"),
            "run_dir": str(run_dir),
            "warnings": [
                "关系研究不自动识别因果；需控制文献、材料体系和实验条件。",
                "XPS 与样本的暂定对应需要核实，分别归一化的 N/O 不提供跨元素化学计量。",
                "原表浓度单位分别保留，不自动把 wt%、w/v%、g/L 合并或把通量重复除以压力。",
                "XPS 推导结构代理量不作为独立结构目标；图片未验证读数不自动进入训练。",
                "比较多个关系/目标后仍需独立验证，不能仅按最高 R² 宣布发现。",
            ],
        }
        prediction_frame.to_csv(run_dir / "predictions.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(metrics).to_csv(
            run_dir / "fold_metrics.csv", index=False, encoding="utf-8-sig"
        )
        (run_dir / "splits.json").write_text(json_dumps(split_manifest), encoding="utf-8")
        (run_dir / "model_selection.json").write_text(json_dumps(selections), encoding="utf-8")
        (run_dir / "summary.json").write_text(json_dumps(result), encoding="utf-8")
        with self.db.connect() as con:
            con.execute(
                "INSERT INTO experiments(experiment_id,hypothesis_id,run_dir,data_hash,split_hash,metrics_json,decision,created_at) VALUES (?,NULL,?,?,?,?,?,?)",
                (
                    run_id,
                    str(run_dir),
                    digest,
                    result["split_hash"],
                    json_dumps(result),
                    "relationship_requires_external_validation",
                    utc_now(),
                ),
            )
        return result
