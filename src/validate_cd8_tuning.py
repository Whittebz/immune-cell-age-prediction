#!/usr/bin/env python3
"""Read-only independent checks of the frozen-cohort CD8 tuning supplement.

No estimator is fitted and no result is written.  ``validate`` returns a JSON-
serializable report; the caller owns persistence.  Prediction reconstruction is
optional because it loads the verified expression matrix into memory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from cell_pipeline import DEFAULT_CONFIG, canonical_digest, read_json, shared_protocol
from preprocess_cd8 import ROOT, sha256

PROTOCOLS = ("donor_cv", "pool_cv")
NEW_MODELS = ("ridge_expanded", "en_expanded", "en_age_weighted")
MODELS = ("median", "ridge_v1", "elastic_net_v1") + NEW_MODELS
AGE_LABELS = ("18-39", "40-59", "60-79", "80-100")
AGE_EDGES = np.array([18., 40., 60., 80., 101.])
ATOL = 1e-9


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError(f"Missing/duplicate CSV columns: {path}")
        rows = list(reader)
    if any(None in row or None in row.values() for row in rows):
        raise ValueError(f"Malformed CSV: {path}")
    return rows


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value, label="number"):
    try:
        result = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid {label}: {value!r}") from exc
    require(np.isfinite(result), f"Nonfinite {label}")
    return result


def integer(value):
    result = number(value, "integer")
    require(result == int(result), f"Not an integer: {value!r}")
    return int(result)


def close(actual, expected, label, atol=ATOL):
    if expected is None:
        require(actual is None or actual == "", f"{label}: expected empty/None")
    elif isinstance(expected, (bool, np.bool_)):
        require(str(actual).lower() == str(bool(expected)).lower(), f"{label}: boolean mismatch")
    else:
        require(np.isclose(number(actual, label), expected, atol=atol, rtol=0),
                f"{label}: {actual!r} != {expected!r}")


def compare_row(actual, expected, label):
    for key, value in expected.items():
        require(key in actual, f"{label}: missing column {key}")
        if isinstance(value, str):
            require(actual[key] == value, f"{label}: {key} mismatch")
        else:
            close(actual[key], value, f"{label}/{key}")


def keyed(rows, fields, expected=None, label="table"):
    result = {}
    for row in rows:
        require(set(fields) <= set(row), f"{label}: missing key columns")
        key = tuple(row[field] for field in fields)
        require(key not in result, f"{label}: duplicate key {key}")
        result[key] = row
    if expected is not None:
        require(set(result) == set(expected), f"{label}: incomplete/unexpected key coverage")
    return result


def age_indices(y):
    y = np.asarray(y, dtype=float)
    require(y.ndim == 1 and np.isfinite(y).all(), "Invalid ages")
    require(np.all((y >= AGE_EDGES[0]) & (y < AGE_EDGES[-1])), "Age outside fixed bins")
    return np.searchsorted(AGE_EDGES, y, side="right") - 1


def regression_metrics(y, prediction, correlation=True):
    """Compute regression metrics without importing the training metric helper."""
    y, prediction = np.asarray(y, float), np.asarray(prediction, float)
    require(y.ndim == 1 and prediction.shape == y.shape, "Prediction/age shape mismatch")
    require(np.isfinite(y).all() and np.isfinite(prediction).all(), "Nonfinite age/prediction")
    n = len(y)
    result = {"n": n, "mae_years": None, "rmse_years": None, "r2": None, "spearman": None}
    if not n:
        return result
    residual = prediction - y
    ssr = float(residual @ residual)
    sst = float(np.sum((y - y.mean()) ** 2))
    result.update(mae_years=float(np.mean(np.abs(residual))), rmse_years=float(np.sqrt(ssr / n)))
    if n > 1:
        result["r2"] = float(1 - ssr / sst) if sst > 0 else float(ssr == 0)
    if correlation and n > 1 and np.ptp(y) > 0 and np.ptp(prediction) > 0:
        a, b = rankdata(y), rankdata(prediction)
        a, b = a - a.mean(), b - b.mean()
        result["spearman"] = float(np.clip((a @ b) / np.sqrt((a @ a) * (b @ b)), -1, 1))
    return result


def macro_age_mae(y, prediction):
    bins = age_indices(y)
    error = np.abs(np.asarray(prediction, float) - np.asarray(y, float))
    require(error.shape == bins.shape and np.isfinite(error).all() and len(error) > 0,
            "Invalid macro-MAE inputs")
    return float(np.mean([error[bins == k].mean() for k in range(4) if np.any(bins == k)]))


def fold_summary(pooled, folds):
    require(len(folds) > 1, "Sample SD requires multiple folds")
    n = sum(integer(row["n"]) for row in folds)
    require(n == integer(pooled["n"]), "Fold sizes do not sum to pooled size")
    result = {"n": n, "n_outer_folds": len(folds)}
    for metric in ("mae_years", "rmse_years", "r2"):
        values = np.array([number(row[metric]) for row in folds])
        result.update({"pooled_" + metric: pooled[metric],
                       "unweighted_fold_mean_" + metric: float(values.mean()),
                       "fold_sample_sd_ddof1_" + metric: float(values.std(ddof=1))})
    weighted = float(np.average([row["mae_years"] for row in folds], weights=[row["n"] for row in folds]))
    delta = weighted - pooled["mae_years"]
    result.update(n_weighted_fold_mae_years=weighted, weighted_minus_pooled_mae=delta,
                  weighted_mae_matches_pooled=bool(abs(delta) <= ATOL))
    return result


def age_weights(y):
    bins = age_indices(y)
    counts = np.bincount(bins, minlength=4)
    require(len(bins) > 0, "Empty training ages")
    weights = counts[bins].astype(float) ** (-0.5)
    return weights / weights.mean(), counts


def weight_audit(ids, ages, train, valid):
    ids, ages = np.asarray(ids, str), np.asarray(ages, float)
    train, valid = np.asarray(train, int), np.asarray(valid, int)
    weights, counts = age_weights(ages[train])
    return {"train_n": len(train), "valid_n": len(valid),
            "train_ids_sha256": canonical_digest(ids[train].tolist()),
            "valid_ids_sha256": canonical_digest(ids[valid].tolist()),
            **{"age_n_" + label.replace("-", "_"): int(count) for label, count in zip(AGE_LABELS, counts)},
            "weight_min": float(weights.min()), "weight_max": float(weights.max()),
            "weight_mean": float(weights.mean()),
            "effective_n": float(weights.sum() ** 2 / (weights @ weights)),
            "weights_sha256": hashlib.sha256(weights.astype("<f8").tobytes()).hexdigest()}


def candidate_key(row):
    model = row["model"]
    require(model in NEW_MODELS, f"Unexpected candidate model: {model}")
    k, alpha = integer(row["max_features"]), number(row["alpha"])
    require(k > 0 and alpha > 0, "Nonpositive hyperparameter")
    l1 = row["l1_ratio"]
    if model == "ridge_expanded":
        require(l1 is None or l1 == "", "Ridge l1_ratio must be empty")
        return k, alpha, None
    l1 = number(l1)
    require(0 < l1 <= 1, "Invalid Elastic Net l1_ratio")
    return k, alpha, l1


def candidate_grid(config):
    """Configuration schema is shared with tune_cd8_common, not inferred from data."""
    for key in ("feature_counts", "ridge_alphas", "elastic_net_alphas", "l1_ratios"):
        values = config[key]
        require(isinstance(values, list) and len(values) > 0, f"Empty/nonlist candidate grid: {key}")
        numeric = [number(value, key) for value in values]
        require(numeric == sorted(set(numeric)) and min(numeric) > 0, f"Unsorted/duplicate/nonpositive grid: {key}")
    require(all(number(k) == integer(k) for k in config["feature_counts"]), "Noninteger feature-count grid")
    require(max(config["l1_ratios"]) <= 1, "Invalid l1_ratio grid")
    return {model: [(int(k), float(alpha), None if model == "ridge_expanded" else float(l1))
                    for k in config["feature_counts"]
                    for alpha in config["ridge_alphas" if model == "ridge_expanded" else "elastic_net_alphas"]
                    for l1 in ([None] if model == "ridge_expanded" else config["l1_ratios"])]
            for model in NEW_MODELS}


def validate_tuning(rows, selected, config, plans):
    """Check full candidate coverage and unweighted global-inner-MAE argmin."""
    grids = candidate_grid(config)
    expected, split_sizes = set(), {}
    for protocol in PROTOCOLS:
        for outer, (train, _) in enumerate(plans[protocol]["outer"]):
            for inner, (itr, iva) in enumerate(plans[protocol]["inner"][outer]):
                split_sizes[protocol, outer, inner] = len(itr), len(iva)
                for model in NEW_MODELS:
                    for candidate in grids[model]:
                        expected.add((protocol, model, outer, inner, *candidate))
    scores = {}
    for row in rows:
        key = (row["protocol"], row["model"], integer(row["outer_fold"]), integer(row["inner_fold"]), *candidate_key(row))
        require(key in expected, f"Unexpected tuning candidate {key}")
        require(key not in scores, f"Duplicate tuning candidate {key}")
        require((integer(row["train_n"]), integer(row["valid_n"])) == split_sizes[key[0], key[2], key[3]],
                "Inner tuning split sizes differ from frozen relative indices")
        mae, macro = number(row["mae_years"]), number(row["macro_age_mae_years"])
        require(mae >= 0 and macro >= 0, "Negative tuning MAE")
        scores[key] = mae
    require(set(scores) == expected, "Incomplete tuning candidate coverage")
    expected_selected = {(p, m, str(o)) for p in PROTOCOLS for m in NEW_MODELS
                         for o in range(len(plans[p]["outer"]))}
    lookup = keyed(selected, ("protocol", "model", "outer_fold"), expected_selected, "selected parameters")
    winners = {}
    for protocol, model, outer_text in sorted(expected_selected):
        outer = int(outer_text)
        count = len(plans[protocol]["inner"][outer])
        means = {candidate: float(np.mean([scores[protocol, model, outer, inner, *candidate]
                                          for inner in range(count)])) for candidate in grids[model]}
        winner = min(means, key=lambda c: (means[c], c[0], c[1], -1 if c[2] is None else c[2]))
        saved = lookup[protocol, model, outer_text]
        require(candidate_key(saved) == winner, f"Selected parameters are not inner global-MAE argmin: {protocol}/{model}/{outer}")
        close(saved["mean_inner_mae_years"], means[winner], "Selected mean inner MAE")
        winners[protocol, model, outer] = winner
    return scores, winners


def validate_split_plan(plans, ids, pools, n_outer, n_inner):
    """Check outer indices and inner indices RELATIVE to each ordered outer train."""
    require(set(PROTOCOLS) <= set(plans) and not (set(plans) - set(PROTOCOLS) - {"demo_inner"}),
            "Unexpected/missing split protocols")
    # common_v1 also freezes a demo_inner plan for its old full-data demo.
    # This supplement must not use it: only the two nested OOF plans are read.
    n = len(ids)
    def check(splits, size, groups, count):
        require(len(splits) == count, "Frozen split count mismatch")
        visits = np.zeros(size, int)
        result = []
        for pair in splits:
            require(len(pair) == 2, "Malformed frozen split")
            train, valid = [np.asarray(part) for part in pair]
            require(all(part.ndim == 1 and np.issubdtype(part.dtype, np.integer) for part in (train, valid)),
                    "Noninteger/nonvector frozen indices")
            require(len(train) > 0 and len(valid) > 0, "Empty frozen split")
            require(set(train) & set(valid) == set() and set(train) | set(valid) == set(range(size))
                    and len(set(train)) == len(train) and len(set(valid)) == len(valid), "Invalid frozen split partition")
            if groups is not None:
                require(not (set(groups[train]) & set(groups[valid])), "Pool leakage in frozen split")
            visits[valid] += 1
            result.append((train.astype(int), valid.astype(int)))
        require(np.all(visits == 1), "Frozen test coverage is not exactly once")
        return result
    normalized = {}
    for protocol in PROTOCOLS:
        groups = np.asarray(pools) if protocol == "pool_cv" else None
        outer = check(plans[protocol]["outer"], n, groups, n_outer)
        require(len(plans[protocol]["inner"]) == n_outer, "Missing inner split plans")
        inner = [check(plans[protocol]["inner"][o], len(train), None if groups is None else groups[train], n_inner)
                 for o, (train, _) in enumerate(outer)]
        normalized[protocol] = {"outer": outer, "inner": inner}
    return normalized


def validate_diagnostics(records, scores, winners, plans, max_iter):
    expected = {("inner_tuning", *key) for key in scores}
    expected |= {("outer_refit", p, m, o, -1, *candidate) for (p, m, o), candidate in winners.items()}
    found, warning_count, outer_warning_count = set(), 0, 0
    for row in records:
        phase = row["phase"]
        p, m, o, i = row["protocol"], row["model"], integer(row["outer_fold"]), integer(row["inner_fold"])
        key = (phase, p, m, o, i, *candidate_key(row))
        require(key in expected and key not in found, "Unexpected/duplicate fit diagnostic")
        found.add(key)
        train_n = len(plans[p]["outer"][o][0]) if phase == "outer_refit" else len(plans[p]["inner"][o][i][0])
        close(row["train_n"], train_n, "Diagnostic training size")
        warnings = row["warnings"]
        require(isinstance(warnings, list), "Malformed fit warnings")
        for warning in warnings:
            require(isinstance(warning, dict) and isinstance(warning.get("category"), str)
                    and isinstance(warning.get("message"), str), "Malformed recorded warning")
        count = sum(w["category"] == "ConvergenceWarning" for w in warnings)
        close(row["convergence_warning_count"], count, "Convergence warning preservation")
        warning_count += count
        outer_warning_count += count if phase == "outer_refit" else 0
        if m == "ridge_expanded":
            require(row["n_iter"] is None and row["dual_gap"] is None, "Ridge solver diagnostics must be null")
        else:
            # sklearn can certify a high-penalty zero solution before its first
            # coordinate-descent sweep, legitimately recording n_iter_=0.
            require(0 <= integer(row["n_iter"]) <= max_iter, "Invalid Elastic Net iteration count")
            number(row["dual_gap"], "dual gap")
    require(found == expected, "Incomplete fit diagnostics; candidates must not be discarded")
    return warning_count, outer_warning_count


def validate_weight_audits(rows, ids, ages, plans):
    expected = {(p, str(o), str(i)) for p in PROTOCOLS for o in range(len(plans[p]["outer"]))
                for i in [-1, *range(len(plans[p]["inner"][o]))]}
    lookup = keyed(rows, ("protocol", "outer_fold", "inner_fold"), expected, "age-weight audit")
    for (p, o_text, i_text), row in lookup.items():
        o, i = int(o_text), int(i_text)
        train, valid = plans[p]["outer"][o]
        if i >= 0:
            itr, iva = plans[p]["inner"][o][i]
            train, valid = train[itr], train[iva]
        compare_row(row, weight_audit(ids, ages, train, valid), "Age-weight audit")


def validate_metric_tables(oof, ids, ages, plans, report):
    n = len(ids)
    expected = {(p, m, str(d)) for p in PROTOCOLS for m in MODELS for d in ids}
    lookup = keyed(oof, ("protocol", "model", "donor_id"), expected, "OOF predictions")
    groups = {(p, m) for p in PROTOCOLS for m in MODELS}
    aggregate = keyed(read_csv(report / "metrics.csv"), ("protocol", "model"), groups, "metrics")
    folds_expected = {(p, m, str(o)) for p, m in groups for o in range(len(plans[p]["outer"]))}
    folds = keyed(read_csv(report / "fold_metrics.csv"), ("protocol", "model", "fold"), folds_expected, "fold metrics")
    bands = keyed(read_csv(report / "age_band_metrics.csv"), ("protocol", "model", "age_band"),
                  {(p, m, band) for p, m in groups for band in AGE_LABELS}, "age-band metrics")
    summaries = keyed(read_csv(report / "fold_summary.csv"), ("protocol", "model"), groups, "fold summary")
    bins = age_indices(ages)
    predictions, measured = {}, []
    for p in PROTOCOLS:
        fold_of = np.empty(n, int)
        for o, (train, test) in enumerate(plans[p]["outer"]):
            fold_of[test] = o
            for index in test:
                close(lookup[p, "median", ids[index]]["predicted_age"], float(np.median(ages[train])), "Training median baseline", 1e-12)
        for model in MODELS:
            pred = np.empty(n)
            for index, donor in enumerate(ids):
                row = lookup[p, model, donor]
                close(row["actual_age"], ages[index], "OOF actual age", 0)
                close(row["fold"], fold_of[index], "OOF frozen fold", 0)
                pred[index] = number(row["predicted_age"])
                close(row["raw_age_gap"], pred[index] - ages[index], "Raw age gap")
            predictions[p, model] = pred
        baseline = np.abs(predictions[p, "median"] - ages)
        for model in MODELS:
            pred = predictions[p, model]
            metrics = regression_metrics(ages, pred, model != "median")
            metrics.update(mae_delta_vs_median_years=float(np.mean(np.abs(pred - ages) - baseline)),
                           macro_age_mae_years=macro_age_mae(ages, pred))
            # Recompute optional descriptive columns when the producer exposes them.
            extra = {"oof_pred_vs_actual_slope": float(np.dot(ages - ages.mean(), pred - pred.mean()) / np.sum((ages - ages.mean()) ** 2)),
                     "oof_mean_signed_residual": float(np.mean(pred - ages)),
                     "mae_delta_vs_v1_en_years": metrics["mae_years"] - float(np.mean(np.abs(predictions[p, "elastic_net_v1"] - ages)))}
            metrics.update({key: value for key, value in extra.items() if key in aggregate[p, model]})
            compare_row(aggregate[p, model], metrics, f"Aggregate metrics/{p}/{model}")
            fold_values = []
            for o, (_, test) in enumerate(plans[p]["outer"]):
                fm = regression_metrics(ages[test], pred[test], model != "median")
                fm.update(mae_delta_vs_median_years=float(np.mean(np.abs(pred[test] - ages[test]) - baseline[test])),
                          macro_age_mae_years=macro_age_mae(ages[test], pred[test]))
                compare_row(folds[p, model, str(o)], fm, "Fold metrics")
                fold_values.append(fm)
            compare_row(summaries[p, model], fold_summary(metrics, fold_values), "Fold mean/sample SD summary")
            for k, label in enumerate(AGE_LABELS):
                mask = bins == k
                bm = regression_metrics(ages[mask], pred[mask], False)
                bm.pop("spearman")
                if mask.sum() < 2 or np.ptp(ages[mask]) == 0:
                    bm["r2"] = None  # Existing age-context schema leaves undefined R² empty.
                bm["signed_residual_mean"] = float(np.mean(pred[mask] - ages[mask])) if np.any(mask) else None
                context = {"age_mean": float(ages[mask].mean()) if np.any(mask) else None,
                           "age_variance_ddof1": float(ages[mask].var(ddof=1)) if mask.sum() > 1 else None,
                           "age_min": float(ages[mask].min()) if np.any(mask) else None,
                           "age_max": float(ages[mask].max()) if np.any(mask) else None}
                bm.update({key: value for key, value in context.items() if key in bands[p, model, label]})
                compare_row(bands[p, model, label], bm, "Age-band metrics")
            measured.append({"protocol": p, "model": model, **metrics})
    return predictions, measured


def validate_reference_oof(predictions, ids, ages, plans):
    main = read_csv(ROOT / "data/models/CD8/common_v1/oof_predictions.csv")
    supplement = read_csv(ROOT / "data/models/CD8/pool_en_common_v1/oof_predictions.csv")
    reference = main + supplement
    rename = {"median": "median", "ridge": "ridge_v1", "elastic_net": "elastic_net_v1"}
    expected = {(p, model, donor) for p in PROTOCOLS for model in ("median", "ridge", "elastic_net") for donor in ids}
    lookup = keyed(reference, ("protocol", "model", "donor_id"), expected, "Original reference OOF")
    index_of = {donor: i for i, donor in enumerate(ids)}
    fold_of = {}
    for p in PROTOCOLS:
        for o, (_, test) in enumerate(plans[p]["outer"]):
            for index in test:
                fold_of[p, ids[index]] = o
    for (p, old, donor), row in lookup.items():
        index = index_of[donor]
        pred = number(row["predicted_age"])
        close(predictions[p, rename[old]][index], pred, "Unchanged reference prediction", 0)
        close(row["actual_age"], ages[index], "Reference actual age", 0)
        close(row["fold"], fold_of[p, donor], "Reference frozen fold", 0)
        close(row["raw_age_gap"], pred - ages[index], "Reference raw age gap")


def validate_outer_archive(path, plans, winners, genes, predictions, ids, X=None):
    labels = [p + "/" + m for p in PROTOCOLS for m in NEW_MODELS]
    n_outer, n_genes = len(plans[PROTOCOLS[0]]["outer"]), len(genes)
    with np.load(path, allow_pickle=False) as saved:
        require(set(saved.files) == {"group_labels", "gene_ids", "coefficients", "available", "means", "scales", "intercepts"},
                "Unexpected numeric model archive schema")
        saved_labels = saved["group_labels"]
        require(saved_labels.ndim == 1 and saved_labels.dtype.kind in "US" and set(saved_labels.tolist()) == set(labels)
                and len(saved_labels) == len(labels), "Numeric model group labels mismatch")
        require(np.array_equal(saved["gene_ids"], np.asarray(genes, str)), "Numeric model gene IDs/order mismatch")
        arrays = {name: saved[name] for name in ("coefficients", "available", "means", "scales", "intercepts")}
        for name, array in arrays.items():
            shape = (6, n_outer) if name == "intercepts" else (6, n_outer, n_genes)
            require(array.shape == shape, f"Numeric model shape mismatch: {name}")
            require(array.dtype.kind in "biuf" and np.isfinite(array).all(), f"Nonfinite/nonnumeric model array: {name}")
        available = arrays["available"]
        require(np.all((available == 0) | (available == 1)), "Invalid feature availability mask")
        available = available.astype(bool)
        require(np.all(arrays["scales"] > 0), "Nonpositive numeric model scales")
        require(np.all(arrays["coefficients"][~available] == 0) and np.all(arrays["means"][~available] == 0)
                and np.all(arrays["scales"][~available] == 1), "Unselected numeric model defaults mismatch")
        by_label = {label: i for i, label in enumerate(saved_labels.tolist())}
        for label, g in by_label.items():
            p, model = label.split("/")
            for o, (train, test) in enumerate(plans[p]["outer"]):
                mask = available[g, o]
                k = winners[p, model, o][0]
                require(0 < mask.sum() <= min(k, n_genes), "Selected feature count exceeds chosen k")
                # Same outer fold + same k must have identical UNWEIGHTED preprocessing.
                for other in NEW_MODELS:
                    h = by_label[p + "/" + other]
                    if winners[p, other, o][0] == k:
                        require(np.array_equal(mask, available[h, o]) and np.array_equal(arrays["means"][g, o], arrays["means"][h, o])
                                and np.array_equal(arrays["scales"][g, o], arrays["scales"][h, o]),
                                "Model strategies do not share unweighted outer preprocessing")
                if X is not None:
                    variance = np.var(X[train], axis=0, dtype=np.float64)
                    eligible = np.flatnonzero(variance > 1e-12)
                    chosen = np.sort(eligible[np.argsort(-variance[eligible], kind="stable")[:k]])
                    require(np.array_equal(np.flatnonzero(mask), chosen), "Outer feature selection not training-only top variance")
                    train_values = np.asarray(X[train][:, chosen], float)
                    mean = train_values.mean(axis=0)
                    scale = np.sqrt(train_values.var(axis=0))
                    scale[scale == 0] = 1
                    require(np.allclose(arrays["means"][g, o, mask], mean, rtol=0, atol=1e-10)
                            and np.allclose(arrays["scales"][g, o, mask], scale, rtol=0, atol=1e-10),
                            "Outer preprocessing differs from unweighted training moments")
                    pred = ((np.asarray(X[test][:, chosen], float) - arrays["means"][g, o, mask])
                            / arrays["scales"][g, o, mask]) @ arrays["coefficients"][g, o, mask] + arrays["intercepts"][g, o]
                    require(np.allclose(pred, predictions[p, model][test], atol=1e-8, rtol=0),
                            "Numeric outer model does not reconstruct saved OOF predictions")


def validate_provenance(manifest, summary, private, shared, frozen, n_donors, n_genes):
    config = manifest["config"]
    require(manifest["config_sha256"] == canonical_digest(config), "Tuning config canonical digest mismatch")
    require(manifest["code_sha256"] == sha256(ROOT / "src/tune_cd8_common.py"), "Tuning code checksum changed")
    close(manifest["n_donors"], n_donors, "Manifest donor count")
    close(manifest["n_input_genes"], n_genes, "Manifest gene count")
    protected = manifest["protected_inputs"]
    require(isinstance(protected, dict) and protected, "Missing protected input checksums")
    core = [ROOT / "data/models/CD8/common_v1" / name for name in
            ("analysis_cohort.csv", "fold_assignments.csv", "oof_predictions.csv")]
    core += [ROOT / "data/models/CD8/pool_en_common_v1/oof_predictions.csv", Path(DEFAULT_CONFIG)]
    core += [shared / name for name in ("protocol.json", "common_cohort.csv", "splits.json")]
    require({str(path.resolve()) for path in core} <= set(protected), "Protected inputs omit frozen core references")
    for name, digest in protected.items():
        path = Path(name)
        require(path.is_absolute(), "Protected input path must be absolute")
        require(isinstance(digest, str) and sha256(path) == digest, f"Protected input changed: {name}")
    expected_source = {"cohort_sha256": frozen["cohort_sha256"], "splits_sha256": frozen["splits_sha256"],
                       "config_canonical_sha256": frozen["config_sha256"],
                       "bundle_sha256": sha256(ROOT / "data/processed/CD8/cd8_pseudobulk.npz")}
    require(manifest["source"] == expected_source and summary["source"] == expected_source,
            "Source provenance does not match frozen source/config/splits")
    require(summary["configuration"] == config, "Summary configuration differs from run manifest")
    close(summary["n_donors"], n_donors, "Summary donor count")
    close(summary["n_input_genes"], n_genes, "Summary gene count")
    artifacts = summary["private_artifacts"]
    required = {"analysis_cohort.csv", "fold_assignments.csv", "oof_predictions.csv", "inner_tuning_scores.csv",
                "age_weight_audit.csv", "fit_diagnostics.json", "outer_models.npz", "run_manifest.json"}
    require(required <= set(artifacts), "Summary omits private artifact hashes")
    for name, artifact in artifacts.items():
        require(Path(name).name == name, "Private artifact must be a basename")
        path = private / name
        require(sha256(path) == artifact["sha256"], f"Private artifact checksum mismatch: {name}")
        close(artifact["bytes"], path.stat().st_size, "Private artifact byte count", 0)


def validate(private: Path, report: Path, shared: Path = ROOT / "data/shared", reconstruct_predictions=False) -> dict:
    """Validate saved artifacts independently, without training or writing files."""
    private, report, shared = Path(private), Path(report), Path(shared)
    manifest = read_json(private / "run_manifest.json")
    config = manifest["config"]
    require(config["protocols"] == list(PROTOCOLS) and config["models"] == list(NEW_MODELS), "Unexpected configured strategies")
    require(config["age_bin_edges"] == AGE_EDGES.tolist() and config["age_weight_power"] == .5, "Age-weight plan changed")
    require(config["tol"] == 1e-4 and integer(config["max_iter"]) > 0, "Invalid solver configuration")
    require(config["full_data_refit"] is False, "Unexpected full-data refit plan")
    candidate_grid(config)
    frozen, shared_cohort, raw_plans = shared_protocol(read_json(DEFAULT_CONFIG), shared)
    # shared_protocol checks the ORIGINAL CONFIG'S CANONICAL digest, never its file bytes.
    cohort_path = ROOT / "data/models/CD8/common_v1/analysis_cohort.csv"
    assignments_path = ROOT / "data/models/CD8/common_v1/fold_assignments.csv"
    for name, source in (("analysis_cohort.csv", cohort_path), ("fold_assignments.csv", assignments_path)):
        require((private / name).read_bytes() == source.read_bytes(), f"Frozen {name} is not an exact copy")
    cohort = read_csv(private / "analysis_cohort.csv")
    ids = [row["donor_id"] for row in cohort]
    require(len(ids) == len(set(ids)) == frozen["n_common_donors"], "Canonical cohort coverage mismatch")
    require(ids == [row["donor_id"] for row in shared_cohort], "Canonical donor order differs from shared index order")
    ages = np.array([number(row["age"]) for row in cohort])
    pools = np.array([row["pool_number"] for row in cohort])
    age_indices(ages)
    metadata = keyed(read_csv(ROOT / "data/processed/CD8/donor_metadata.csv"), ("donor_id",), label="Source metadata")
    for row in cohort:
        original = metadata[row["donor_id"],]
        require(all(row[key] == original[key] for key in original), "Analysis metadata differs from prepared source")
    plans = validate_split_plan(raw_plans, ids, pools, config["outer_folds"], config["inner_folds"])
    assignments = keyed(read_csv(private / "fold_assignments.csv"), ("protocol", "donor_id"),
                        {(p, d) for p in PROTOCOLS for d in ids}, "Fold assignments")
    for p in PROTOCOLS:
        for o, (_, test) in enumerate(plans[p]["outer"]):
            for i in test:
                compare_row(assignments[p, ids[i]], {"fold": o, "pool_number": str(pools[i])}, "Frozen fold assignments")
    predictions, recomputed = validate_metric_tables(read_csv(private / "oof_predictions.csv"), ids, ages, plans, report)
    validate_reference_oof(predictions, ids, ages, plans)
    scores, winners = validate_tuning(read_csv(private / "inner_tuning_scores.csv"),
                                      read_csv(report / "selected_parameters.csv"), config, plans)
    validate_weight_audits(read_csv(private / "age_weight_audit.csv"), ids, ages, plans)
    records = read_json(private / "fit_diagnostics.json")
    all_warnings, outer_warnings = validate_diagnostics(records, scores, winners, plans, config["max_iter"])
    genes = [row["gene_id"] for row in read_csv(ROOT / "data/processed/CD8/gene_metadata.csv")]
    X = None
    if reconstruct_predictions:
        from run_cell_supplement import inputs as frozen_inputs
        _, X, source_y, source_ids, source_genes, _, _, _, _, _, _ = frozen_inputs("CD8", "common", read_json(DEFAULT_CONFIG), shared)
        require(source_ids.tolist() == ids and np.array_equal(source_y, ages), "Reconstruction source donor/age order mismatch")
        require(source_genes.tolist() == genes, "Reconstruction source gene order mismatch")
    validate_outer_archive(private / "outer_models.npz", plans, winners, genes, predictions, ids, X)
    summary = read_json(report / "summary.json")
    validate_provenance(manifest, summary, private, shared, frozen, len(ids), len(genes))
    close(summary["n_pools"], len(set(pools)), "Summary pool count")
    close(summary["fit_count"], len(records), "Summary fit count")
    close(summary["convergence_warning_count_all_fits"], all_warnings, "Summary all convergence warnings")
    close(summary["convergence_warning_count_outer_refits"], outer_warnings, "Summary outer convergence warnings")
    expected_groups = {(p, m) for p in PROTOCOLS for m in NEW_MODELS}
    convergence = keyed(summary["convergence_by_strategy"], ("protocol", "model"), expected_groups, "Summary convergence groups")
    convergence_csv = (keyed(read_csv(report / "convergence_summary.csv"), ("protocol", "model"), expected_groups, "Convergence table")
                       if (report / "convergence_summary.csv").exists() else None)
    for p, m in expected_groups:
        part = [row for row in records if row["protocol"] == p and row["model"] == m]
        expected = {"fits": len(part),
                    "inner_warning_messages": sum(row["convergence_warning_count"] for row in part if row["phase"] == "inner_tuning"),
                    "outer_warning_messages": sum(row["convergence_warning_count"] for row in part if row["phase"] == "outer_refit")}
        compare_row(convergence[p, m], expected, "Summary strategy convergence")
        if convergence_csv is not None:
            compare_row(convergence_csv[p, m], expected, "Strategy convergence table")
    summary_metrics = keyed(summary["metrics"], ("protocol", "model"),
                            {(p, m) for p in PROTOCOLS for m in MODELS}, "Summary metrics")
    for row in recomputed:
        compare_row(summary_metrics[row["protocol"], row["model"]], row, "Summary metrics")
    csv_selected = read_csv(report / "selected_parameters.csv")
    json_selected = [dict(row, outer_fold=str(integer(row["outer_fold"]))) for row in summary["selected_parameters"]]
    selected_lookup = keyed(json_selected, ("protocol", "model", "outer_fold"),
                            {(r["protocol"], r["model"], r["outer_fold"]) for r in csv_selected}, "Summary selected parameters")
    for row in csv_selected:
        saved = selected_lookup[row["protocol"], row["model"], row["outer_fold"]]
        require(candidate_key(saved) == candidate_key(row), "Summary selected parameters mismatch")
        close(saved["mean_inner_mae_years"], number(row["mean_inner_mae_years"]), "Summary inner MAE")
    result = {"status": "passed", "n_donors": len(ids), "oof_rows_checked": len(ids) * 12,
              "inner_candidate_fits_checked": len(scores), "outer_refits_checked": len(winners),
              "fit_diagnostics_checked": len(records), "age_weight_audits_checked": sum(len(plans[p]["outer"]) * (1 + config["inner_folds"]) for p in PROTOCOLS),
              "convergence_warnings_all_fits": all_warnings, "convergence_warnings_outer_refits": outer_warnings,
              "frozen_cohort_and_splits_verified": True, "reference_predictions_unchanged": True,
              "aggregate_fold_age_metrics_recomputed": True, "inner_global_mae_selection_verified": True,
              "age_weights_reconstructed": True, "numeric_outer_models_verified": True,
              "predictions_reconstructed": bool(reconstruct_predictions), "recomputed_metrics": recomputed,
              "scope": "Saved-artifact consistency and optional source-expression prediction reconstruction; no training, external-cohort or biological validation."}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--shared", type=Path, default=ROOT / "data/shared")
    parser.add_argument("--reconstruct-predictions", action="store_true")
    args = parser.parse_args()
    print(json.dumps(validate(args.private, args.report, args.shared, args.reconstruct_predictions), indent=2, allow_nan=False))
