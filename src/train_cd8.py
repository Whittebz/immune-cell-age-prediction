#!/usr/bin/env python3
"""Nested CPU age-regression benchmark on validated donor-level CD8 data.

No actual age, identifiers, pool or cell-composition features enter gene models.
Inner preprocessing is reused across hyperparameters only within the same split.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
from pathlib import Path
import platform
import time
import warnings
from datetime import datetime, timezone

import numpy as np
import scipy
from scipy.stats import spearmanr
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted
from threadpoolctl import threadpool_limits

from preprocess_cd8 import ROOT, sha256, write_csv

MODEL_NAMES = ("median", "ridge", "elastic_net")
GRIDS = {
    "ridge": {"alpha": [0.1, 1.0, 10.0, 100.0]},
    "elastic_net": {"alpha": [0.01, 0.1, 1.0], "l1_ratio": [0.1, 0.5, 0.9]},
}


class TopVarianceGenes(TransformerMixin, BaseEstimator):
    """Select nonconstant genes by variance of ONLY the rows passed to fit."""
    def __init__(self, max_features=1000):
        self.max_features = max_features

    def fit(self, X, y=None):
        X = np.asarray(X)
        if X.ndim != 2 or self.max_features < 1 or not np.isfinite(X).all():
            raise ValueError("Expected finite 2-D input and a positive feature limit")
        self.n_features_in_ = X.shape[1]
        variances = np.var(X, axis=0, dtype=np.float64)
        eligible = np.flatnonzero(variances > 1e-12)
        if not eligible.size:
            raise ValueError("No nonconstant training genes")
        order = np.argsort(-variances[eligible], kind="stable")
        self.indices_ = np.sort(eligible[order[:self.max_features]])
        return self

    def transform(self, X):
        check_is_fitted(self, "indices_")
        if np.ndim(X) != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("Feature dimensions differ from training")
        return np.asarray(X[:, self.indices_], dtype=np.float64)


def make_preprocessor(max_features):
    return Pipeline([("genes", TopVarianceGenes(max_features)), ("scale", StandardScaler())])


def build_regressor(name, params, max_iter=50000, seed=42):
    if name == "ridge":
        return Ridge(**params, solver="cholesky")
    if name == "elastic_net":
        return ElasticNet(**params, max_iter=max_iter, tol=1e-4,
                          selection="cyclic", random_state=seed)
    raise ValueError(f"Unknown model {name}")


def fit_recorded(model, X, y, diagnostics, context):
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        model.fit(X, y)
    estimator = model.named_steps["regressor"] if isinstance(model, Pipeline) else model
    record = {
        **context, "n_iter": int(np.max(estimator.n_iter_)) if getattr(estimator, "n_iter_", None) is not None else None,
        "dual_gap": float(np.max(estimator.dual_gap_)) if getattr(estimator, "dual_gap_", None) is not None else None,
        "convergence_warning_count": sum(issubclass(w.category, ConvergenceWarning) for w in records),
        "warnings": [{"category": w.category.__name__, "message": str(w.message)} for w in records],
    }
    diagnostics.append(record)
    return model


def make_splits(X, y, groups, folds, seed, grouped=False):
    if grouped:
        if len(np.unique(groups)) < folds:
            raise ValueError("Insufficient distinct pools for grouped CV")
        splitter = GroupKFold(n_splits=folds, shuffle=True, random_state=seed)
        splits = list(splitter.split(X, y, groups))
    else:
        splits = list(KFold(n_splits=folds, shuffle=True, random_state=seed).split(X))
    visits = np.zeros(len(y), dtype=int)
    for train, test in splits:
        if set(train) & set(test):
            raise AssertionError("Training/testing overlap")
        if grouped and set(groups[train]) & set(groups[test]):
            raise AssertionError("Training/testing pool overlap")
        visits[test] += 1
    if not np.all(visits == 1):
        raise AssertionError("Not exactly one test-fold assignment per donor")
    return splits


def validate_fixed_splits(splits, n_rows, groups=None, expected_folds=None):
    """Validate frozen split indices; each test row occurs once, no leakage."""
    if expected_folds is not None and len(splits) != expected_folds:
        raise ValueError("Frozen split count differs from protocol")
    visits = np.zeros(n_rows, dtype=int)
    result = []
    universe = set(range(n_rows))
    for train, test in splits:
        train, test = np.asarray(train, dtype=int), np.asarray(test, dtype=int)
        if train.ndim != 1 or test.ndim != 1 or not len(train) or not len(test):
            raise ValueError("Frozen split is empty or not one-dimensional")
        tr, te = set(train.tolist()), set(test.tolist())
        if len(tr) != len(train) or len(te) != len(test) or tr & te or tr | te != universe:
            raise ValueError("Frozen train/test split is invalid")
        if groups is not None and set(groups[train]) & set(groups[test]):
            raise ValueError("Frozen grouped split leaks pools")
        visits[test] += 1
        result.append((train, test))
    if not np.all(visits == 1):
        raise ValueError("Frozen test-fold coverage is not exactly once")
    return result


def tune_models(X, y, splits, names, max_features, diagnostics, context,
                max_iter=50000, seed=42, tuning_rows=None):
    """Explicit inner CV: fit selector/scaler fresh on each INNER training set.

    Reuse that split's transformed arrays across candidates, avoiding repeated
    high-dimensional preprocessing without using any validation rows to fit it.
    """
    candidates = {name: list(ParameterGrid(GRIDS[name])) for name in names}
    scores = {name: [[] for _ in candidates[name]] for name in names}
    for inner_fold, (train, valid) in enumerate(splits):
        prep = make_preprocessor(max_features)
        Ztrain = prep.fit_transform(X[train], y[train])
        Zvalid = prep.transform(X[valid])
        for name in names:
            for candidate_index, params in enumerate(candidates[name]):
                reg = build_regressor(name, params, max_iter, seed)
                fit_recorded(reg, Ztrain, y[train], diagnostics, {
                    **context, "phase": "inner_tuning", "inner_fold": inner_fold,
                    "model": name, "params": params,
                })
                error = mean_absolute_error(y[valid], reg.predict(Zvalid))
                scores[name][candidate_index].append(float(error))
                if tuning_rows is not None:
                    tuning_rows.append({**context, "inner_fold": inner_fold, "model": name,
                                        "params": json.dumps(params, sort_keys=True), "mae_years": float(error)})
    # Deterministic tie-breaking follows ParameterGrid order; no outer score used.
    return {name: candidates[name][int(np.argmin([np.mean(s) for s in scores[name]]))] for name in names}


def metrics(y, prediction, correlation=True):
    result = {
        "n": int(len(y)), "mae_years": float(mean_absolute_error(y, prediction)),
        "rmse_years": float(np.sqrt(mean_squared_error(y, prediction))),
        "r2": float(r2_score(y, prediction)), "spearman": None,
    }
    if correlation and np.std(y) > 0 and np.std(prediction) > 0:
        result["spearman"] = float(spearmanr(y, prediction).statistic)
    return result


def save_numeric_model(path, pipeline, gene_ids):
    """A non-pickle demo artifact; its refit is not an independent benchmark."""
    np.savez_compressed(
        path, gene_ids=np.asarray(gene_ids, dtype=str),
        feature_indices=pipeline.named_steps["genes"].indices_,
        mean=pipeline.named_steps["scale"].mean_, scale=pipeline.named_steps["scale"].scale_,
        coefficient=pipeline.named_steps["regressor"].coef_,
        intercept=np.asarray(pipeline.named_steps["regressor"].intercept_),
    )


def predict_numeric_model(X, gene_ids, path):
    with np.load(path, allow_pickle=False) as saved:
        if not np.array_equal(gene_ids, saved["gene_ids"]):
            raise ValueError("Gene IDs/order must exactly match the model input schema")
        selected = np.asarray(X[:, saved["feature_indices"]], dtype=float)
        return ((selected - saved["mean"]) / saved["scale"]) @ saved["coefficient"] + saved["intercept"]


def feature_stability(coefficient_vectors, availability_vectors, gene_ids, symbols):
    coefficients = np.stack(coefficient_vectors)
    availability = np.stack(availability_vectors)
    nonzero = np.abs(coefficients) > 1e-10
    rows = []
    for g in np.flatnonzero(availability.any(axis=0)):
        values = coefficients[:, g][nonzero[:, g]]
        signs = np.sign(values)
        direction = float(max(np.sum(signs > 0), np.sum(signs < 0)) / len(values)) if len(values) else 0.0
        rows.append({
            "gene_id": str(gene_ids[g]), "gene_symbol": str(symbols[g]),
            "available_folds": int(availability[:, g].sum()), "nonzero_folds": int(len(values)),
            "nonzero_given_available": float(len(values) / availability[:, g].sum()),
            "sign_consistency_when_nonzero": direction,
            "mean_abs_standardized_coefficient": float(np.mean(np.abs(coefficients[:, g]))),
            "mean_signed_standardized_coefficient": float(np.mean(coefficients[:, g])),
            "stable_by_rule": int(len(values) >= int(np.ceil(0.8 * len(coefficient_vectors))) and direction >= 0.8),
        })
    return sorted(rows, key=lambda r: (-r["stable_by_rule"], -r["nonzero_folds"],
                                      -r["mean_abs_standardized_coefficient"], r["gene_id"]))


def export_figure(fig, path):
    fig.savefig(path, dpi=160)
    buffer = io.StringIO()
    fig.savefig(buffer, format="svg")
    path.with_suffix(".svg").write_text("\n".join(line.rstrip() for line in buffer.getvalue().splitlines()) + "\n", encoding="utf-8")


def plots(report, y, predictions, aggregate, fold_rows, stability, outer_folds=5, cell_label="CD8"):
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    colors = {"median": "#969b9f", "ridge": "#287c8e", "elastic_net": "#c77c41"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for ax, name in zip(axes, ("ridge", "elastic_net")):
        pred = predictions["donor_cv"][name]
        score = next(r for r in aggregate if r["protocol"] == "donor_cv" and r["model"] == name)
        ax.scatter(y, pred, s=12, alpha=.45, color=colors[name], edgecolors="none")
        ax.plot([15, 100], [15, 100], "--", color="#555555", linewidth=1)
        ax.set(xlabel="Actual age (years)", ylabel="Out-of-fold prediction (years)",
               title=f"{name} | MAE {score['mae_years']:.2f} y | R2 {score['r2']:.2f}")
    fig.suptitle(f"{cell_label} gene-expression age prediction | donor-level nested CV")
    export_figure(fig, report / "oof_predictions.png")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for ax, protocol, title in zip(axes, ("donor_cv", "pool_cv"), ("Donor CV (shared pools allowed)", "Pool-held-out sensitivity")):
        rows = [r for r in aggregate if r["protocol"] == protocol]
        ax.bar([r["model"] for r in rows], [r["mae_years"] for r in rows], color=[colors[r["model"]] for r in rows], alpha=.85)
        for index, row in enumerate(rows):
            errors = [r["mae_years"] for r in fold_rows if r["protocol"] == protocol and r["model"] == row["model"]]
            ax.scatter(np.full(len(errors), index), errors, color="#30343b", s=16, zorder=3)
        ax.set(ylabel="MAE (years; lower is better)", title=title)
    fig.suptitle("Bars: pooled OOF MAE | Dots: fold MAE, not confidence intervals")
    export_figure(fig, report / "benchmark.png")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, name in zip(axes, ("ridge", "elastic_net")):
        residual = predictions["donor_cv"][name] - y
        ax.scatter(y, residual, s=10, alpha=.35, color=colors[name], edgecolors="none")
        ax.axhline(0, color="#555555", linestyle="--")
        ax.set(xlabel="Actual age (years)", ylabel="Predicted minus actual age (years)", title=name)
    fig.suptitle("Raw residuals are prediction errors, not proven biological age acceleration")
    export_figure(fig, report / "age_residuals.png")
    plt.close(fig)
    selected = [r for r in stability if r["stable_by_rule"]][:10]
    stable_plot = bool(selected)
    if not selected:
        selected = [r for r in stability if r["nonzero_folds"]][:10]
    if selected:
        fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
        positions = np.arange(len(selected))
        values = [r["mean_signed_standardized_coefficient"] for r in selected]
        ax.barh(positions, values, color=["#c77c41" if v > 0 else "#287c8e" for v in values])
        ax.set_yticks(positions, [f"{r['gene_symbol']} ({r['nonzero_folds']}/{outer_folds})" for r in selected])
        ax.invert_yaxis()
        ax.set(xlabel="Mean standardized coefficient across outer folds (years)",
               title=("Elastic Net predictive features | correlated genes are not causal drivers" if stable_plot
                      else "Exploratory features only | none passed the stability rule"))
        export_figure(fig, report / "feature_stability.png")
        plt.close(fig)


def load_data(data_dir, preprocessing_summary, bundle_name="cd8_pseudobulk.npz"):
    source = json.loads(preprocessing_summary.read_text(encoding="utf-8"))
    for name in (bundle_name, "donor_metadata.csv"):
        if sha256(data_dir / name) != source["artifacts"][name]["sha256"]:
            raise ValueError(f"Preprocessing provenance checksum mismatch: {name}")
    with np.load(data_dir / bundle_name, allow_pickle=False) as saved:
        X, ids, genes, symbols = saved["log1p_cpm"], saved["donor_ids"], saved["gene_ids"], saved["gene_symbols"]
    with (data_dir / "donor_metadata.csv").open(newline="", encoding="utf-8") as stream:
        meta = list(csv.DictReader(stream))
    if ids.tolist() != [r["donor_id"] for r in meta] or len(np.unique(ids)) != len(ids):
        raise ValueError("Donor IDs must be unique and exactly aligned")
    y = np.array([float(r["age"]) for r in meta])
    groups = np.array([r["pool_number"] for r in meta])
    if X.shape != (len(ids), len(genes)) or not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("Invalid expression/target dimensions or nonfinite data")
    return X, y, ids, genes, symbols, groups, source


def run_benchmark(X, y, ids, genes, symbols, groups, output, report,
                  seed=42, max_features=1000, outer_folds=5, inner_folds=3,
                  max_iter=50000, make_plots=True, source=None, cell_label="CD8",
                  fixed_outer_splits=None, fixed_inner_splits=None, demo_inner_splits=None,
                  cohort_records=None, run_metadata=None):
    output, report = Path(output).resolve(), Path(report).resolve()
    for path in (output, report):
        if path.exists() and any(path.iterdir()):
            raise FileExistsError(f"Refusing to overwrite {path}")
    if output == report or output in report.parents or report in output.parents:
        raise ValueError("Private and aggregate output directories must be separate")
    if len(np.unique(ids)) != len(y) or X.shape != (len(y), len(genes)):
        raise ValueError("Misaligned/nonunique donor data")
    output.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    diagnostics, tuning_rows, fold_rows, oof_rows, assignment_rows, aggregate, age_rows = [], [], [], [], [], [], []
    coefficient_vectors, availability_vectors = [], []
    predictions = {}
    with threadpool_limits(limits=2):
        for protocol, grouped, names in (("donor_cv", False, MODEL_NAMES), ("pool_cv", True, ("median", "ridge"))):
            predictions[protocol] = {name: np.full(len(y), np.nan) for name in names}
            if fixed_outer_splits is None:
                outer = make_splits(X, y, groups, outer_folds, seed, grouped)
            else:
                outer = validate_fixed_splits(fixed_outer_splits[protocol], len(y),
                                              groups if grouped else None, outer_folds)
            for fold, (train, test) in enumerate(outer):
                print(f"[{protocol}] outer fold {fold + 1}/{outer_folds}: train={len(train)}, test={len(test)}", flush=True)
                for index in test:
                    assignment_rows.append({"protocol": protocol, "donor_id": str(ids[index]), "fold": fold,
                                            "pool_number": str(groups[index])})
                Xtrain, Xtest, ytrain = X[train], X[test], y[train]
                if fixed_inner_splits is None:
                    inner = make_splits(Xtrain, ytrain, groups[train], inner_folds, seed + fold + 1, grouped)
                else:
                    inner = validate_fixed_splits(fixed_inner_splits[protocol][fold], len(train),
                                                  groups[train] if grouped else None, inner_folds)
                learned = [name for name in names if name != "median"]
                best = tune_models(Xtrain, ytrain, inner, learned, max_features, diagnostics,
                                   {"protocol": protocol, "outer_fold": fold}, max_iter, seed, tuning_rows)
                baseline = DummyRegressor(strategy="median").fit(np.zeros((len(train), 1)), ytrain)
                for name in names:
                    if name == "median":
                        pred = baseline.predict(np.zeros((len(test), 1)))
                        selected_params, n_genes = {}, 0
                    else:
                        pipeline = Pipeline([("genes", TopVarianceGenes(max_features)), ("scale", StandardScaler()),
                                             ("regressor", build_regressor(name, best[name], max_iter, seed))])
                        fit_recorded(pipeline, Xtrain, ytrain, diagnostics, {
                            "protocol": protocol, "outer_fold": fold, "phase": "outer_refit",
                            "model": name, "params": best[name],
                        })
                        pred = pipeline.predict(Xtest)
                        selected_params, n_genes = best[name], len(pipeline.named_steps["genes"].indices_)
                        if protocol == "donor_cv" and name == "elastic_net":
                            coefficients = np.zeros(X.shape[1]); available = np.zeros(X.shape[1], dtype=bool)
                            selected = pipeline.named_steps["genes"].indices_
                            coefficients[selected] = pipeline.named_steps["regressor"].coef_
                            available[selected] = True
                            coefficient_vectors.append(coefficients)
                            availability_vectors.append(available)
                    predictions[protocol][name][test] = pred
                    scores = metrics(y[test], pred, name != "median")
                    fold_rows.append({"protocol": protocol, "model": name, "fold": fold,
                                      **scores, "train_n": len(train), "train_pool_n": len(np.unique(groups[train])),
                                      "test_pool_n": len(np.unique(groups[test])),
                                      "test_age_min": float(y[test].min()), "test_age_max": float(y[test].max()),
                                      "selected_genes": n_genes, "parameters": json.dumps(selected_params, sort_keys=True)})
                    print(f"  {name}: MAE={scores['mae_years']:.3f} years, params={selected_params}", flush=True)
                    for index, value in zip(test, pred):
                        oof_rows.append({"protocol": protocol, "model": name, "donor_id": str(ids[index]),
                                         "fold": fold, "actual_age": float(y[index]), "predicted_age": float(value),
                                         "raw_age_gap": float(value - y[index])})
            baseline_error = np.abs(predictions[protocol]["median"] - y)
            for name in names:
                pred = predictions[protocol][name]
                if not np.isfinite(pred).all():
                    raise ValueError("Incomplete/nonfinite OOF predictions")
                score = {"protocol": protocol, "model": name, **metrics(y, pred, name != "median"),
                         "mae_delta_vs_median_years": float(np.mean(np.abs(pred - y) - baseline_error))}
                aggregate.append(score)
                for low, high in ((18, 40), (40, 60), (60, 80), (80, 101)):
                    subset = (y >= low) & (y < high)
                    if subset.sum() >= 2:
                        age_rows.append({"protocol": protocol, "model": name, "age_band": f"{low}-{high - 1}",
                                         **metrics(y[subset], pred[subset], name != "median")})
        # Prespecified Elastic Net full-cohort demo refit; not another test result.
        print("Refitting prespecified Elastic Net demo artifact on all donors (not a benchmark)...", flush=True)
        inner = (make_splits(X, y, groups, inner_folds, seed + 100, False) if demo_inner_splits is None
                 else validate_fixed_splits(demo_inner_splits, len(y), expected_folds=inner_folds))
        final_params = tune_models(X, y, inner, ["elastic_net"], max_features, diagnostics,
                                   {"protocol": "full_data_demo", "outer_fold": -1}, max_iter, seed, tuning_rows)["elastic_net"]
        final = Pipeline([("genes", TopVarianceGenes(max_features)), ("scale", StandardScaler()),
                          ("regressor", build_regressor("elastic_net", final_params, max_iter, seed))])
        fit_recorded(final, X, y, diagnostics, {"protocol": "full_data_demo", "outer_fold": -1,
                                              "phase": "full_refit", "model": "elastic_net", "params": final_params})
    model_path = output / "elastic_net_demo_model.npz"
    save_numeric_model(model_path, final, genes)
    np.testing.assert_allclose(final.predict(X[:5]), predict_numeric_model(X[:5], genes, model_path), rtol=1e-10, atol=1e-10)
    stability = feature_stability(coefficient_vectors, availability_vectors, genes, symbols)
    if cohort_records is not None:
        if [r["donor_id"] for r in cohort_records] != ids.tolist():
            raise ValueError("Analysis-cohort records are not aligned")
        write_csv(output / "analysis_cohort.csv", list(cohort_records[0]), cohort_records)
    write_csv(output / "oof_predictions.csv", list(oof_rows[0]), oof_rows)
    write_csv(output / "fold_assignments.csv", list(assignment_rows[0]), assignment_rows)
    write_csv(output / "inner_tuning_scores.csv", list(tuning_rows[0]), tuning_rows)
    (output / "fit_diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8")
    write_csv(report / "metrics.csv", list(aggregate[0]), aggregate)
    write_csv(report / "fold_metrics.csv", list(fold_rows[0]), fold_rows)
    write_csv(report / "age_band_metrics.csv", list(age_rows[0]), age_rows)
    write_csv(report / "elastic_net_feature_stability.csv", list(stability[0]), stability)
    if make_plots:
        plots(report, y, predictions, aggregate, fold_rows, stability, outer_folds, cell_label)
    convergence = sum(r["convergence_warning_count"] for r in diagnostics)
    winning_warnings = sum(r["convergence_warning_count"] for r in diagnostics if r["phase"] in ("outer_refit", "full_refit"))
    summary = {
        "status": ("completed_with_refit_convergence_warnings" if winning_warnings
                   else "completed_with_tuning_convergence_warnings" if convergence else "completed"),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cell_label": cell_label, "run_metadata": run_metadata or {},
        "n_donors": len(y), "n_input_genes": X.shape[1], "n_pools": len(np.unique(groups)),
        "source_preprocessing_sha256": source["input"]["sha256"] if source else None,
        "configuration": {"seed": seed, "max_features": max_features, "outer_folds": outer_folds,
                          "inner_folds": inner_folds, "max_iter": max_iter, "tol": 1e-4,
                          "grids": GRIDS, "blas_threads": 2, "main_model": "elastic_net",
                          "input_features": "gene expression only", "age_correction": None},
        "metrics": aggregate, "convergence_warning_count_all_fits": convergence,
        "convergence_warning_count_outer_or_full_refits": winning_warnings,
        "n_recorded_fits": len(diagnostics), "demo_model_parameters": final_params,
        "demo_model_roundtrip_verified": True,
        "stable_en_genes_by_rule": sum(r["stable_by_rule"] for r in stability),
        "stability_rule": f"nonzero in >={int(np.ceil(0.8 * outer_folds))}/{outer_folds} donor-CV outer folds; sign consistency >=0.8 conditional on nonzero",
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__, "sklearn": sklearn.__version__},
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "limitations": [
            "Chronological-age prediction is not independently validated biological immune age.",
            "Donor CV allows shared pools; only Ridge and median have additional held-out-pool sensitivity.",
            f"Single cohort, older-skewed ages, broad-{cell_label} subtype composition and technical effects remain relevant.",
            "OOF raw age gaps include prediction errors and age-related regression-to-mean; not calibrated acceleration.",
            "Five overlapping training folds do not constitute independent replication of feature biology.",
            "Full-data demo refit is not an additional performance evaluation; no training score is reported.",
        ],
        "private_artifacts": {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size} for p in sorted(output.iterdir())},
    }
    lines = [f"# {cell_label} 年龄回归初步实验\n", "实际年龄为监督目标；所有性能来自外层折外预测，不是训练集分数。\n",
             "| 验证 | 模型 | MAE（年） | RMSE（年） | R² | Spearman |", "|---|---|---:|---:|---:|---:|"]
    for r in aggregate:
        rank = "不适用" if r["spearman"] is None else f"{r['spearman']:.3f}"
        lines.append(f"| {r['protocol']} | {r['model']} | {r['mae_years']:.3f} | {r['rmse_years']:.3f} | {r['r2']:.3f} | {rank} |")
    lines += ["\n## 验证与收敛\n", f"供者级外层 {outer_folds} 折、内层 {inner_folds} 折；每个内层训练折单独筛选高方差基因和标准化。",
              "同一内层划分的预处理只在不同参数候选间复用，不跨训练/验证边界。",
              "pool 留出仅验证 Ridge 和中位数基线，不能据此声称 Elastic Net 已通过新批次验证。",
              f"全部拟合中的收敛警告数：{convergence}；外层/全数据重拟合中的收敛警告数：{winning_warnings}。",
              "具体参数、迭代次数、dual gap 和警告上下文保存在本地 fit_diagnostics.json；不静默丢弃候选。",
              "\n## 特征与限制\n", f"按预定规则，Elastic Net 有 {summary['stable_en_genes_by_rule']} 个相对稳定特征。",
              "区分高方差筛选可用频次、非零系数频次和符号一致性；相关基因可能相互替代。",
              "这些是预测相关特征，不是独立验证的衰老因果基因；文献证据须在单独的生物学解释环节核实。",
              "实际年龄分布偏老；原始年龄差同时包含预测误差，不据此诊断个体加速衰老。",
              "总体 MAE 改善不保证所有年龄段均优于基线；分段结果见 age_band_metrics.csv。",
              "全数据 Elastic Net 重拟合产物只供后续演示，其拟合数据不能作为独立测试。",
              f"\n逐供者预测和模型数值位于本地 {output}；本报告仅给汇总结果，请勿公开逐供者文件。\n"]
    (report / "REPORT.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    (report / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(aggregate, indent=2), flush=True)
    print(f"Completed benchmark in {summary['elapsed_seconds']}s; summary: {report / 'summary.json'}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/processed/CD8")
    parser.add_argument("--preprocessing-summary", type=Path, default=ROOT / "results/cd8/summary.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/models/CD8")
    parser.add_argument("--report", type=Path, default=ROOT / "results/cd8_training")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-features", type=int, default=1000)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to((ROOT / "data/models").resolve()):
        parser.error("Participant-level outputs must remain inside the gitignored data/models directory")
    X, y, ids, genes, symbols, groups, source = load_data(args.data, args.preprocessing_summary)
    run_benchmark(X, y, ids, genes, symbols, groups, args.output, args.report,
                  seed=args.seed, max_features=args.max_features, make_plots=not args.no_plots, source=source)


if __name__ == "__main__":
    main()
