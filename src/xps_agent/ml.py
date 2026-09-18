from __future__ import annotations

import itertools
import json
import os
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import Settings
from .db import StateDB
from .features import SPECTRUM_SUMMARY_VERSION, is_known_descriptor, is_quality_column
from .utils import json_dumps, sha256_file, sha256_text, utc_now


@dataclass
class FoldMetric:
    feature_set: str
    outer_fold: int
    model: str
    n_train: int
    n_test: int
    train_groups: int
    test_groups: int
    r2: float
    mae: float
    rmse: float
    spearman: float


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    correlation = (
        spearmanr(y_true, y_pred, nan_policy="omit").statistic
        if len(y_true) > 1 and np.unique(y_true).size > 1 and np.unique(y_pred).size > 1
        else float("nan")
    )
    return {
        "r2": float(r2_score(y_true, y_pred))
        if len(y_true) >= 2 and np.unique(y_true).size > 1
        else float("nan"),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "spearman": float(correlation) if np.isfinite(correlation) else float("nan"),
    }


def _sign_permutation_pvalue(deltas: np.ndarray) -> float:
    deltas = deltas[np.isfinite(deltas)]
    if not len(deltas):
        return float("nan")
    observed = float(np.mean(deltas))
    if len(deltas) <= 15:
        values = []
        for signs in itertools.product((-1.0, 1.0), repeat=len(deltas)):
            values.append(float(np.mean(deltas * np.asarray(signs))))
    else:
        rng = np.random.default_rng(20260915)
        values = [
            float(np.mean(deltas * rng.choice((-1.0, 1.0), len(deltas)))) for _ in range(10000)
        ]
    hits = sum(value >= observed for value in values)
    return float(hits / len(values)) if len(deltas) <= 15 else float((hits + 1) / (len(values) + 1))


def prepare_evaluation_data(
    frame: pd.DataFrame,
    target: str,
    group: str,
    baseline: list[str],
    candidates: list[str],
    *,
    candidate_policy: str = "measured_only",
    include_unresolved_groups: bool = False,
    dataset_scope: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if any(name.startswith("_input_") for name in [target, group, *baseline, *candidates]):
        raise ValueError("Internal provenance columns cannot be selected as model variables")
    if target == group or target in baseline + candidates or group in baseline + candidates:
        raise ValueError("Target/group columns cannot be predictors (direct leakage)")
    if not baseline or not candidates:
        raise ValueError("Baseline and candidate feature lists must both be non-empty")
    if len(set(baseline)) != len(baseline) or len(set(candidates)) != len(candidates):
        raise ValueError("Feature lists must not contain duplicate columns")
    if set(baseline) & set(candidates):
        raise ValueError("Candidate features must add columns not already in the baseline")
    if candidate_policy not in {"measured_only", "impute"}:
        raise ValueError("candidate_policy must be measured_only or impute")
    required = list(dict.fromkeys([target, group, *baseline, *candidates]))
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError(f"Columns not found: {missing}")
    data = frame[required].copy()
    data["_input_row_index"] = np.arange(len(frame))
    if "record_id" in frame:
        data["_input_record_id"] = frame["record_id"].astype(str)
    for column in [target, *baseline, *candidates]:
        data[column] = pd.to_numeric(data[column], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
    group_text = data[group].astype("string").str.strip()
    valid_group = group_text.notna() & ~group_text.str.lower().isin(
        {"", "nan", "none", "nat", "inf", "-inf"}
    )
    eligible = data[target].notna() & valid_group
    dropped_dataset = 0
    if dataset_scope is not None:
        if not dataset_scope or "dataset" not in frame:
            raise ValueError("A non-empty dataset scope requires the dataset column")
        selected = frame["dataset"].astype(str).isin(dataset_scope)
        dropped_dataset = int((eligible & ~selected).sum())
        eligible &= selected
    dropped_unresolved = 0
    if "doi_group" in frame:
        doi = frame["doi_group"].fillna("").astype(str).str.strip()
        known = doi != ""
        grouping = pd.DataFrame({"doi": doi[known], "group": group_text[known]})
        if (grouping.groupby("doi")["group"].nunique() > 1).any():
            raise ValueError(
                "One DOI is split across multiple selected groups; resolve grouping first"
            )
        if not include_unresolved_groups:
            dropped_unresolved = int((eligible & ~known).sum())
            eligible &= known
    dropped_missing_candidates = 0
    if candidate_policy == "measured_only":
        measured = data[candidates].notna().all(axis=1)
        dropped_missing_candidates = int((eligible & ~measured).sum())
        eligible &= measured
    data[group] = group_text
    report = {
        "input_rows": len(frame),
        "rows_scored": int(eligible.sum()),
        "dropped_missing_or_nonfinite_target_or_group": int(
            (~(data[target].notna() & valid_group)).sum()
        ),
        "dropped_unresolved_doi": dropped_unresolved,
        "dropped_missing_candidates": dropped_missing_candidates,
        "candidate_policy": candidate_policy,
        "include_unresolved_groups": include_unresolved_groups,
        "dataset_scope": dataset_scope,
        "dropped_outside_dataset_scope": dropped_dataset,
    }
    return data.loc[eligible].reset_index(drop=True), report


class NestedGroupEvaluator:
    def __init__(self, settings: Settings, db: StateDB):
        self.settings = settings
        self.db = db

    @staticmethod
    def _pipeline_and_grid(seed: int) -> tuple[Pipeline, list[dict[str, Any]]]:
        pipeline = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scale", StandardScaler()),
                ("model", ElasticNet(max_iter=20000, random_state=seed)),
            ]
        )
        grid = [
            {
                "model": [ElasticNet(max_iter=20000, random_state=seed)],
                "model__alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
                "model__l1_ratio": [0.05, 0.25, 0.5, 0.75, 0.95],
            },
            {
                "model": [
                    RandomForestRegressor(
                        n_estimators=400,
                        random_state=seed,
                        n_jobs=1,
                        max_features="sqrt",
                    )
                ],
                "model__max_depth": [None, 4, 8],
                "model__min_samples_leaf": [1, 2, 4],
            },
        ]
        return pipeline, grid

    def evaluate(
        self,
        table: Path,
        target: str,
        group: str,
        baseline: list[str],
        candidates: list[str],
        outer_splits: int = 5,
        inner_splits: int = 4,
        seed: int = 20260915,
        hypothesis_id: str | None = None,
        candidate_policy: str = "measured_only",
        include_unresolved_groups: bool = False,
        dataset_scope: list[str] | None = None,
    ) -> dict[str, Any]:
        if hypothesis_id and not self.db.rows(
            "SELECT hypothesis_id FROM hypotheses WHERE hypothesis_id=?", (hypothesis_id,)
        ):
            raise ValueError("Unknown hypothesis_id; refusing an unlinked validation run")
        if table.suffix.lower() in {".xlsx", ".xls"}:
            frame = pd.read_excel(table)
        else:
            frame = pd.read_csv(table)
        if outer_splits < 3 or inner_splits < 2:
            raise ValueError("At least 3 outer and 2 inner grouped folds are required")
        data_hash = sha256_file(table)
        if (
            table.resolve()
            == (self.settings.workspace_root / "canonical" / "model_table_enriched.csv").resolve()
        ):
            try:
                manifest = json.loads(
                    table.with_suffix(".manifest.json").read_text(encoding="utf-8")
                )
                source = self.settings.workspace_root / "canonical" / "model_table.csv"
                if (
                    manifest.get("source_sha256") != sha256_file(source)
                    or manifest.get("destination_sha256") != data_hash
                    or manifest.get("spectrum_summary_version") != SPECTRUM_SUMMARY_VERSION
                ):
                    raise ValueError(
                        "Enriched table is stale; re-materialize descriptors before ML"
                    )
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(
                    "Enriched table requires a valid materialization manifest"
                ) from exc
        inventory_path = self.settings.workspace_root / "canonical" / "column_inventory.csv"
        if inventory_path.exists():
            inventory = pd.read_csv(inventory_path)
            forbidden = set(
                inventory.loc[
                    inventory["role"].isin(["outcome", "identity", "spectrum_quality"]), "column"
                ]
            )
            leakage = [
                name
                for name in baseline + candidates
                if name in forbidden or is_quality_column(name)
            ]
            leakage.extend(name for name in candidates if is_known_descriptor(name))
            if leakage:
                raise ValueError(
                    f"Outcome/identity/known/quality-artifact predictors are forbidden: {leakage}"
                )
        data, cohort = prepare_evaluation_data(
            frame,
            target,
            group,
            baseline,
            candidates,
            candidate_policy=candidate_policy,
            include_unresolved_groups=include_unresolved_groups,
            dataset_scope=dataset_scope,
        )
        groups = data[group].astype(str)
        group_count = groups.nunique()
        if group_count < 3:
            raise ValueError("At least three independent groups are required")
        n_outer = min(outer_splits, group_count)
        outer_cv = GroupKFold(n_splits=n_outer)
        split_indices = list(outer_cv.split(data, data[target], groups))
        feature_sets = {
            "baseline": baseline,
            "augmented": list(dict.fromkeys([*baseline, *candidates])),
        }
        fold_metrics: list[FoldMetric] = []
        predictions: list[dict[str, Any]] = []
        split_manifest: list[dict[str, Any]] = []
        selection_manifest: list[dict[str, Any]] = []
        dummy_predictions: list[dict[str, Any]] = []
        for outer_fold, (train_index, test_index) in enumerate(split_indices, start=1):
            train_groups = groups.iloc[train_index]
            test_groups = groups.iloc[test_index]
            overlap = set(train_groups) & set(test_groups)
            if overlap:
                raise RuntimeError(f"Group leakage detected: {sorted(overlap)[:3]}")
            split_manifest.append(
                {
                    "outer_fold": outer_fold,
                    "train_groups": sorted(set(train_groups)),
                    "test_groups": sorted(set(test_groups)),
                    "train_input_rows": data.iloc[train_index]["_input_row_index"].tolist(),
                    "test_input_rows": data.iloc[test_index]["_input_row_index"].tolist(),
                }
            )
            training_mean = float(data.loc[train_index, target].mean())
            dummy_predictions.extend(
                {"observed": float(data.loc[index, target]), "predicted": training_mean}
                for index in test_index
            )
            for feature_set, features in feature_sets.items():
                if not features:
                    raise ValueError("The baseline feature list cannot be empty")
                x_train = data.loc[train_index, features]
                x_test = data.loc[test_index, features]
                y_train = data.loc[train_index, target].to_numpy()
                y_test = data.loc[test_index, target].to_numpy()
                empty_training_features = x_train.columns[x_train.isna().all()].tolist()
                if empty_training_features:
                    raise ValueError(
                        f"Outer fold {outer_fold} has no training measurements for: {empty_training_features}"
                    )
                inner_group_count = train_groups.nunique()
                if inner_group_count < 2:
                    raise ValueError("Each outer training fold needs at least two groups")
                inner_cv = GroupKFold(n_splits=min(inner_splits, inner_group_count))
                pipeline, grid = self._pipeline_and_grid(seed + outer_fold)
                search = GridSearchCV(
                    pipeline,
                    grid,
                    scoring="neg_mean_absolute_error",
                    cv=inner_cv,
                    n_jobs=min(4, os.cpu_count() or 1),
                    refit=True,
                    error_score="raise",
                )
                search.fit(x_train, y_train, groups=train_groups)
                predicted = search.predict(x_test)
                metric = _metrics(y_test, predicted)
                model_name = type(search.best_estimator_.named_steps["model"]).__name__
                selection_manifest.append(
                    {
                        "outer_fold": outer_fold,
                        "feature_set": feature_set,
                        "selected_model": model_name,
                        "best_inner_mae": -float(search.best_score_),
                        "parameters": {
                            key: str(value) for key, value in search.best_params_.items()
                        },
                    }
                )
                fold_metrics.append(
                    FoldMetric(
                        feature_set=feature_set,
                        outer_fold=outer_fold,
                        model=model_name,
                        n_train=len(train_index),
                        n_test=len(test_index),
                        train_groups=train_groups.nunique(),
                        test_groups=test_groups.nunique(),
                        **metric,
                    )
                )
                for row_index, observed, estimate, group_value in zip(
                    test_index,
                    y_test,
                    predicted,
                    test_groups,
                    strict=True,
                ):
                    predictions.append(
                        {
                            "row_index": int(row_index),
                            "input_row_index": int(data.loc[row_index, "_input_row_index"]),
                            "record_id": data.loc[row_index, "_input_record_id"]
                            if "_input_record_id" in data
                            else None,
                            "group": str(group_value),
                            "outer_fold": outer_fold,
                            "feature_set": feature_set,
                            "observed": float(observed),
                            "predicted": float(estimate),
                        }
                    )
        metric_frame = pd.DataFrame([asdict(item) for item in fold_metrics])
        summary_by_set = (
            metric_frame.groupby("feature_set")[["r2", "mae", "rmse", "spearman"]]
            .agg(["mean", "std", "median"])
            .to_dict()
        )
        paired = metric_frame.pivot(index="outer_fold", columns="feature_set", values="r2")
        deltas = (paired["augmented"] - paired["baseline"]).to_numpy()
        prediction_frame = pd.DataFrame(predictions)
        pooled = {
            name: _metrics(part["observed"].to_numpy(), part["predicted"].to_numpy())
            for name, part in prediction_frame.groupby("feature_set")
        }
        group_balanced = {}
        for name, part in prediction_frame.groupby("feature_set"):
            per_group = [
                _metrics(item["observed"].to_numpy(), item["predicted"].to_numpy())
                for _, item in part.groupby("group")
            ]
            group_balanced[name] = {
                metric: float(np.mean([item[metric] for item in per_group]))
                for metric in ("mae", "rmse")
            }
        dummy = pd.DataFrame(dummy_predictions)
        if sha256_file(table) != data_hash:
            raise RuntimeError("Input table changed during evaluation; rerun on a frozen version")
        split_hash = sha256_text(json_dumps(split_manifest))
        run_key = sha256_text(
            json_dumps(
                {
                    "table": str(table.resolve()),
                    "data_hash": data_hash,
                    "target": target,
                    "group": group,
                    "baseline": baseline,
                    "candidates": candidates,
                    "split_hash": split_hash,
                    "seed": seed,
                    "cohort": cohort,
                }
            )
        )[:12]
        timestamp = utc_now().replace(":", "").replace("+", "_")
        run_dir = (
            self.settings.workspace_root
            / "runs"
            / f"ml_{timestamp}_{run_key}_{uuid.uuid4().hex[:6]}"
        )
        run_dir.mkdir(parents=True, exist_ok=False)
        metric_frame.to_csv(run_dir / "fold_metrics.csv", index=False, encoding="utf-8-sig")
        prediction_frame.to_csv(run_dir / "predictions.csv", index=False, encoding="utf-8-sig")
        (run_dir / "splits.json").write_text(json_dumps(split_manifest), encoding="utf-8")
        (run_dir / "model_selection.json").write_text(
            json_dumps(selection_manifest), encoding="utf-8"
        )
        previous_runs = self.db.rows(
            "SELECT COUNT(*) count FROM experiments WHERE data_hash=?", (data_hash,)
        )[0]["count"]
        result = {
            "created_at": utc_now(),
            "table": str(table.resolve()),
            "data_hash": data_hash,
            "evaluator_code_sha256": sha256_file(Path(__file__)),
            "seed": seed,
            "outer_splits": n_outer,
            "inner_splits": inner_splits,
            "cohort": cohort,
            "pooled_out_of_fold": pooled,
            "group_balanced_out_of_fold": group_balanced,
            "training_mean_dummy_out_of_fold": _metrics(
                dummy["observed"].to_numpy(), dummy["predicted"].to_numpy()
            ),
            "prior_runs_same_input": previous_runs,
            "rows_scored": len(data),
            "groups": group_count,
            "target": target,
            "group": group,
            "baseline": baseline,
            "candidates": candidates,
            "split_hash": split_hash,
            "summary": {str(key): value for key, value in summary_by_set.items()},
            "delta_r2_by_fold": deltas.tolist(),
            "mean_delta_r2": float(np.mean(deltas[np.isfinite(deltas)]))
            if np.isfinite(deltas).any()
            else float("nan"),
            "one_sided_sign_permutation_p": _sign_permutation_pvalue(deltas),
            "claim_policy": "Tentative ranking only; requires mechanism, counterexample, and external validation.",
            "inference_warning": "Fold sign-flip p is exploratory only: outer training sets overlap, candidates/runs may be adaptively selected. It is not a valid discovery significance test or multiple-testing correction.",
            "linkage_warning": "Row-order spectrum links and unresolved DOI groups must be reviewed before physical claims.",
            "run_dir": str(run_dir),
        }
        (run_dir / "summary.json").write_text(json_dumps(result), encoding="utf-8")
        experiment_id = sha256_text(str(run_dir))[:24]
        with self.db.connect() as con:
            con.execute(
                """
                INSERT INTO experiments
                (experiment_id, hypothesis_id, run_dir, data_hash, split_hash,
                 metrics_json, decision, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    experiment_id,
                    hypothesis_id,
                    str(run_dir),
                    result["data_hash"],
                    split_hash,
                    json_dumps(result),
                    "pending_evidence_review",
                    utc_now(),
                ),
            )
        return result
