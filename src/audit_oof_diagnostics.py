#!/usr/bin/env python3
"""Read-only, aggregate-only diagnostics of frozen out-of-fold age predictions.

Example (use the project's existing virtualenv):
  .venv/bin/python src/audit_oof_diagnostics.py --cell CD8
  ... --extra-oof results/supplement/oof_predictions.csv

No fitting of prediction models, recalibration, participant-level output or
changes to source artifacts. Extra OOF files must cover the SAME complete
cohort, use existing folds, and add disjoint protocol/model/donor keys. A model's
analysis_cohort.csv, when present, declares its genuinely trained cohort; every
row must match source preparation metadata and training/summary.json n_donors.
OOF rows are NEVER subsetted. Output must be new or empty (historical audits
are not overwritten). Use --cell B, CD4, CD8, etc. with explicit input paths.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.preprocessing import SplineTransformer

ROOT = Path(__file__).resolve().parents[1]
METRIC_NAMES = ("mae_years", "rmse_years", "r2")
BANDS = ("30-59", "60-119", "120-239", ">=240")
ATOL = RTOL = 1e-9


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        raise ValueError("Refusing to write an empty aggregate table")
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if isinstance(v, (float, np.floating)) and not np.isfinite(v) else v)
                             for k, v in row.items()})


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_dir(path, cell, filename, training=False):
    path = Path(path).resolve()
    candidates = [path, path / cell]
    if training:
        candidates.append(path / (cell.lower() + "_training"))
    for candidate in candidates:
        if (candidate / filename).is_file():
            return candidate
    raise FileNotFoundError(f"Cannot locate {filename} under {path}")


def regression_metrics(y, prediction):
    y, prediction = np.asarray(y, float), np.asarray(prediction, float)
    if len(y) == 0:
        return {"n": 0, "mae_years": np.nan, "rmse_years": np.nan, "r2": np.nan}
    error = prediction - y
    sst = np.sum((y - y.mean()) ** 2)
    return {"n": len(y), "mae_years": float(np.mean(np.abs(error))),
            "rmse_years": float(np.sqrt(np.mean(error ** 2))),
            "r2": float(1 - np.sum(error ** 2) / sst) if len(y) > 1 and sst > 0 else np.nan}


def context(y, prediction):
    y = np.asarray(y, float)
    result = regression_metrics(y, prediction)
    result.update(age_mean=float(y.mean()) if len(y) else np.nan,
                  age_variance_ddof1=float(y.var(ddof=1)) if len(y) > 1 else np.nan,
                  age_min=float(y.min()) if len(y) else np.nan,
                  age_max=float(y.max()) if len(y) else np.nan,
                  signed_residual_mean=float(np.mean(np.asarray(prediction) - y)) if len(y) else np.nan)
    return result


def cell_band(value):
    if value < 30:
        return "<30"
    if value < 60:
        return "30-59"
    if value < 120:
        return "60-119"
    if value < 240:
        return "120-239"
    return ">=240"


def depth_quartiles(values):
    """Cohort-wide numerical cutpoints; tied depths are never split arbitrarily."""
    cuts = np.quantile(np.asarray(values, float), [0.25, 0.5, 0.75])
    return np.searchsorted(cuts, values, side="left") + 1, cuts


def nuisance_design(age, sex, pool):
    """Nonlinear age basis + sex/pool fixed effects, not a predictive refit."""
    age = np.asarray(age, float)
    basis = SplineTransformer(n_knots=5, degree=3, knots="quantile", include_bias=False)
    if len(np.unique(age)) >= 5:
        spline = basis.fit_transform(age[:, None])
    else:
        z = (age - age.mean()) / max(float(age.std()), 1.0)
        spline = np.column_stack([z, z ** 2, z ** 3])
    columns = [np.ones((len(age), 1)), spline]
    for values in (np.asarray(sex, str), np.asarray(pool, str)):
        levels = sorted(set(values))
        columns.append(np.column_stack([values == level for level in levels[1:]])
                       if len(levels) > 1 else np.empty((len(age), 0)))
    return np.column_stack(columns)


def rank_association(x, y, design):
    """Spearman and Pearson correlation of separately residualized average ranks.

    Orthogonal projection by SVD handles aliased pool/sex/age columns. No p-values:
    donor clustering, multiple diagnostics and estimated nuisance terms preclude
    interpreting naive iid tests as confirmatory evidence.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    result = {"n": len(x), "raw_spearman": np.nan, "adjusted_rank_residual_r": np.nan,
              "nuisance_rank": 0, "residual_df": 0, "adjusted_status": "insufficient_or_constant"}
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return result
    result["raw_spearman"] = float(spearmanr(x, y).statistic)
    q, s, _ = np.linalg.svd(design, full_matrices=False)
    rank = int(np.sum(s > (max(design.shape) * np.finfo(float).eps * s[0])))
    result.update(nuisance_rank=rank, residual_df=len(x) - rank)
    if len(x) - rank < 3:
        return result
    q = q[:, :rank]
    rx, ry = rankdata(x), rankdata(y)
    ex, ey = rx - q @ (q.T @ rx), ry - q @ (q.T @ ry)
    if np.linalg.norm(ex) <= 1e-10 * np.linalg.norm(rx) or np.linalg.norm(ey) <= 1e-10 * np.linalg.norm(ry):
        result["adjusted_status"] = "constant_residual_ranks"
        return result
    result["adjusted_rank_residual_r"] = float(np.clip(np.dot(ex, ey) / np.linalg.norm(ex) / np.linalg.norm(ey), -1, 1))
    result["adjusted_status"] = "ok_descriptive_not_causal"
    return result


def cell_fields(cell):
    return f"n_{cell.lower()}_cells", f"actual_{cell.lower()}_umi"


def select_analysis_metadata(metadata, analysis_cohort=None, summary=None, cell="CD8"):
    """Validate an explicitly declared trained cohort, never slice OOF predictions.

    Full preparation metadata may exceed a genuinely retrained common-cohort
    model. Every declared row must exactly match its preparation source values.
    """
    count_key, depth_key = cell_fields(cell)
    fields = ("donor_id", "age", "sex", "pool_number", count_key, depth_key)
    source = {}
    for row in metadata:
        if not set(fields).issubset(row):
            raise ValueError("Metadata schema missing required columns")
        if row["donor_id"] in source:
            raise ValueError("Duplicate donor in source preparation metadata")
        source[row["donor_id"]] = row
    selected = [row for row in metadata if str(row.get("included", "1")).lower() in ("1", "true", "1.0")]
    if analysis_cohort is not None:
        if summary is None:
            raise ValueError("Declared analysis cohort requires training/summary.json n_donors")
        if not analysis_cohort:
            raise ValueError("Declared analysis cohort is empty")
        selected, seen = [], set()
        for row in analysis_cohort:
            if not set(fields).issubset(row):
                raise ValueError("Analysis cohort schema missing required metadata columns")
            donor = row["donor_id"]
            if donor in seen or donor not in source:
                raise ValueError("Analysis cohort IDs must be unique subset of source metadata")
            seen.add(donor)
            original = source[donor]
            if str(original.get("included", "1")).lower() not in ("1", "true", "1.0"):
                raise ValueError("Analysis cohort includes source-excluded donor")
            for field in ("age", count_key, depth_key):
                if not np.isfinite([float(row[field]), float(original[field])]).all() or float(row[field]) != float(original[field]):
                    raise ValueError("Analysis cohort numeric metadata differs from preparation source")
            if any(row[field] != original[field] for field in ("sex", "pool_number")):
                raise ValueError("Analysis cohort sex/pool differs from preparation source")
            selected.append({**original, **row})
    if summary is not None and ("n_donors" not in summary or float(summary["n_donors"]) != len(selected)):
        raise ValueError("Training summary n_donors differs from declared analysis cohort")
    if not selected:
        raise ValueError("No included analysis donors")
    return selected


def prepare_output(output):
    output = Path(output).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Output must be a new or empty directory; refusing nonempty output")
    return output


def validate_and_join(oof_sources, metadata, assignments, cell="CD8"):
    count_key, depth_key = cell_fields(cell)
    required = {"donor_id", "age", "sex", "pool_number", count_key, depth_key}
    if not metadata or not required.issubset(metadata[0]):
        raise ValueError("Metadata schema missing required columns")
    meta = {}
    for row in metadata:
        if row["donor_id"] in meta:
            raise ValueError("Duplicate donor in metadata")
        if str(row.get("included", "1")).lower() not in ("1", "true", "1.0"):
            continue
        meta[row["donor_id"]] = row
    folds = {}
    for row in assignments:
        key = (row["protocol"], row["donor_id"])
        if key in folds:
            raise ValueError("Duplicate donor/protocol fold assignment")
        folds[key] = (int(row["fold"]), str(row["pool_number"]))
    groups, seen, base_keys = {}, set(), set()
    for source_index, source in enumerate(oof_sources):
        for row in source:
            required_oof = {"protocol", "model", "donor_id", "fold", "actual_age", "predicted_age"}
            if not required_oof.issubset(row):
                raise ValueError("OOF schema missing required columns")
            key = (row["protocol"], row["model"], row["donor_id"])
            if key in seen:
                raise ValueError("Duplicate protocol/model/donor across OOF inputs")
            seen.add(key)
            if source_index == 0:
                base_keys.add(key[:2])
            if row["donor_id"] not in meta:
                raise ValueError("OOF includes unknown/excluded metadata donor")
            donor = meta[row["donor_id"]]
            assignment = folds.get((row["protocol"], row["donor_id"]))
            if assignment is None or assignment != (int(row["fold"]), str(donor["pool_number"])):
                raise ValueError("OOF fold/pool differs from frozen assignment")
            y, pred = float(row["actual_age"]), float(row["predicted_age"])
            cells, depth = float(donor[count_key]), float(donor[depth_key])
            if not np.isfinite([y, pred, float(donor["age"]), cells, depth]).all() or cells <= 0 or depth <= 0:
                raise ValueError("Nonfinite age/prediction or nonpositive cell count/UMI")
            if not donor["sex"] or not donor["pool_number"]:
                raise ValueError("Missing sex/pool in metadata")
            if not np.isclose(y, float(donor["age"]), atol=ATOL, rtol=RTOL):
                raise ValueError("OOF actual age differs from metadata")
            if row.get("raw_age_gap") and not np.isclose(float(row["raw_age_gap"]), pred - y, atol=ATOL, rtol=RTOL):
                raise ValueError("OOF signed residual differs from predicted minus actual age")
            groups.setdefault(key[:2], []).append({"donor_id": row["donor_id"], "fold": int(row["fold"]),
                "age": y, "pred": pred, "sex": donor["sex"], "pool": donor["pool_number"],
                "cells": cells, "depth": depth})
    for key, rows in groups.items():
        if {r["donor_id"] for r in rows} != set(meta):
            raise ValueError("Each protocol/model must cover the complete included metadata cohort; no OOF slicing")
        if {r["fold"] for r in rows} != set(range(5)):
            raise ValueError("Expected exactly five outer folds numbered 0 through 4")
        if key[0] == "pool_cv":
            pools = {}
            for row in rows:
                pools.setdefault(row["pool"], set()).add(row["fold"])
            if any(len(v) != 1 for v in pools.values()):
                raise ValueError("Pool appears in multiple pool-CV test folds")
    if not groups:
        raise ValueError("No OOF groups")
    return groups, base_keys


def compare_source_metrics(pooled, folds, source_pooled, source_folds, base_keys):
    comparisons = []
    for kind, actual, source, keys in (("pooled", pooled, source_pooled, ("protocol", "model")),
                                       ("fold", folds, source_folds, ("protocol", "model", "fold"))):
        lookup = {tuple(str(r[k]) for k in keys): r for r in actual}
        source_seen = set()
        for row in source:
            key = tuple(str(row[k]) for k in keys)
            if key in source_seen or key not in lookup:
                raise ValueError("Source metric duplicate or missing corresponding OOF coverage")
            source_seen.add(key)
            for metric in ("n",) + METRIC_NAMES + ("spearman", "mae_delta_vs_median_years"):
                if row.get(metric, "") == "":
                    continue
                observed, expected = float(lookup[key][metric]), float(row[metric])
                ok = bool(np.isclose(observed, expected, atol=ATOL, rtol=RTOL, equal_nan=True))
                comparisons.append({"scope": kind, "protocol": row["protocol"], "model": row["model"],
                                    "fold": row.get("fold", ""), "metric": metric, "recomputed": observed,
                                    "source": expected, "abs_difference": abs(observed - expected), "passed": ok})
        expected_keys = {key for key in lookup if key[:2] in base_keys}
        if not expected_keys.issubset(source_seen):
            raise ValueError("Frozen base OOF group missing source metric rows")
    return comparisons


def compute_tables(groups, cell="CD8"):
    count_key, depth_key = cell_fields(cell)
    pooled, fold_rows, fold_summary, sex_rows, assoc_rows, band_rows, age_rows = ([] for _ in range(7))
    first = next(iter(groups.values()))
    depth_map = {row["donor_id"]: row["depth"] for row in first}
    quartiles, cuts = depth_quartiles(list(depth_map.values()))
    qmap = dict(zip(depth_map, quartiles))
    for (protocol, model), rows in sorted(groups.items()):
        label = {"protocol": protocol, "model": model}
        y, pred, folds = (np.array([r[c] for r in rows]) for c in ("age", "pred", "fold"))
        sex, pool = (np.array([r[c] for r in rows]) for c in ("sex", "pool"))
        residual = pred - y
        metrics = regression_metrics(y, pred)
        metrics["spearman"] = float(spearmanr(y, pred).statistic) if model != "median" and np.ptp(pred) else np.nan
        baseline = groups.get((protocol, "median"))
        baseline_mae = np.mean([abs(r["pred"] - r["age"]) for r in baseline]) if baseline else np.nan
        metrics["mae_delta_vs_median_years"] = metrics["mae_years"] - baseline_mae
        pooled.append({**label, **metrics})
        current_folds = []
        for fold in range(5):
            keep = folds == fold
            item = {**label, "fold": fold, **regression_metrics(y[keep], pred[keep])}
            item["spearman"] = float(spearmanr(y[keep], pred[keep]).statistic) if model != "median" and np.ptp(pred[keep]) else np.nan
            current_folds.append(item)
            fold_rows.append(item)
        summary = {**label, "n": len(y), "n_outer_folds": 5}
        for metric in METRIC_NAMES:
            values = [r[metric] for r in current_folds]
            summary["pooled_" + metric] = metrics[metric]
            summary["unweighted_fold_mean_" + metric] = float(np.mean(values))
            summary["fold_sample_sd_ddof1_" + metric] = float(np.std(values, ddof=1))
        weighted = float(np.average([r["mae_years"] for r in current_folds], weights=[r["n"] for r in current_folds]))
        summary.update(n_weighted_fold_mae_years=weighted, weighted_minus_pooled_mae=weighted - metrics["mae_years"],
                       weighted_mae_matches_pooled=bool(np.isclose(weighted, metrics["mae_years"], atol=ATOL, rtol=RTOL)))
        fold_summary.append(summary)
        for level in sorted(set(sex)):
            keep = sex == level
            sex_rows.append({**label, "sex": level, **context(y[keep], pred[keep])})
        design = nuisance_design(y, sex, pool)
        for exposure in ("cells", "depth"):
            x = np.array([r[exposure] for r in rows])
            for outcome, values in (("absolute_oof_error", np.abs(residual)), ("signed_residual_pred_minus_actual", residual)):
                assoc_rows.append({**label, "exposure": count_key if exposure == "cells" else depth_key,
                    "outcome": outcome, **rank_association(x, values, design),
                    "adjustment": "average_ranks_residualized_on_age_quantile_cubic_bspline_5knots_plus_sex_pool_fixed_effects"})
        band_arrays = [("cell_count", np.array([cell_band(r["cells"]) for r in rows]), list(BANDS)),
                       ("actual_umi_quartile", np.array([f"Q{qmap[r['donor_id']]}" for r in rows]), ["Q1", "Q2", "Q3", "Q4"])]
        if any(r["cells"] < 30 for r in rows):
            band_arrays[0][2].insert(0, "<30")
        for stratifier, values, levels in band_arrays:
            for level in levels:
                keep = values == level
                selected = [r for r, take in zip(rows, keep) if take]
                band_rows.append({**label, "stratifier": stratifier, "band": level,
                    **context(y[keep], pred[keep]),
                    "female_n": sum(r["sex"] == "female" for r in selected),
                    "cell_count_min": min((r["cells"] for r in selected), default=np.nan),
                    "cell_count_max": max((r["cells"] for r in selected), default=np.nan),
                    "actual_umi_min": min((r["depth"] for r in selected), default=np.nan),
                    "actual_umi_max": max((r["depth"] for r in selected), default=np.nan)})
        for low, high in ((18, 40), (40, 60), (60, 80), (80, 101)):
            keep = (y >= low) & (y < high)
            age_rows.append({**label, "age_band": f"{low}-{high-1}", **context(y[keep], pred[keep])})
    return {"pooled_metrics": pooled, "fold_metrics_recomputed": fold_rows, "fold_summary": fold_summary,
            "sex_stratified_metrics": sex_rows, "error_depth_associations": assoc_rows,
            "cell_depth_band_metrics": band_rows, "age_context_metrics": age_rows}, cuts


def make_plots(tables, output):
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    summaries = tables["fold_summary"]
    labels = [f"{r['protocol']} / {r['model']}" for r in summaries]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    x = np.arange(len(summaries))
    axes[0].errorbar(x, [r["unweighted_fold_mean_mae_years"] for r in summaries],
                     yerr=[r["fold_sample_sd_ddof1_mae_years"] for r in summaries], fmt="o", capsize=4,
                     label="Unweighted 5-fold mean +/- sample SD (not CI)")
    axes[0].scatter(x, [r["pooled_mae_years"] for r in summaries], marker="x", label="Pooled OOF MAE", zorder=4)
    axes[0].set(ylabel="MAE (years)", title="Pooled vs fold summary")
    axes[0].set_xticks(x, labels, rotation=35, ha="right")
    axes[0].legend(fontsize=7)
    for sex, offset in (("female", -0.15), ("male", 0.15)):
        values = [next(r["mae_years"] for r in tables["sex_stratified_metrics"]
                       if r["protocol"] == s["protocol"] and r["model"] == s["model"] and r["sex"] == sex)
                  if any(r["protocol"] == s["protocol"] and r["model"] == s["model"] and r["sex"] == sex
                         for r in tables["sex_stratified_metrics"]) else np.nan for s in summaries]
        axes[1].bar(x + offset, values, width=0.3, label=sex)
    axes[1].set_xticks(x, labels, rotation=35, ha="right")
    axes[1].set(ylabel="MAE (years)", title="Sex strata (descriptive; age mix differs)")
    axes[1].legend()
    fig.tight_layout()
    for extension in ("png", "svg"):
        destination = output / f"aggregate_oof_metrics.{extension}"
        fig.savefig(destination, dpi=170)
        if extension == "svg":
            destination.write_text("\n".join(line.rstrip() for line in destination.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    exposures = list(dict.fromkeys(r["exposure"] for r in tables["error_depth_associations"]))
    for axis, (exposure, outcome) in zip(axes.flat, [(e, o) for e in exposures
                                                    for o in ("absolute_oof_error", "signed_residual_pred_minus_actual")]):
        rows = [r for r in tables["error_depth_associations"] if r["exposure"] == exposure and r["outcome"] == outcome]
        axis.bar(x - .16, [r["raw_spearman"] for r in rows], width=.32, label="Raw Spearman")
        axis.bar(x + .16, [r["adjusted_rank_residual_r"] for r in rows], width=.32, label="Age spline / sex / pool adjusted")
        axis.axhline(0, color="black", linewidth=.6)
        axis.set_xticks(x, labels, rotation=35, ha="right", fontsize=8)
        axis.set(title=f"{exposure}: {outcome.replace('_', ' ')}", ylabel="Rank association (not causal)", ylim=(-1, 1))
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    for extension in ("png", "svg"):
        destination = output / f"aggregate_rank_associations.{extension}"
        fig.savefig(destination, dpi=170)
        if extension == "svg":
            destination.write_text("\n".join(line.rstrip() for line in destination.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
    plt.close(fig)


def report(tables, status, output):
    n = tables["pooled_metrics"][0]["n"]
    links = {name: f"[{name}](<{output / name}>)" for name in
             ("cell_depth_band_metrics.csv", "age_context_metrics.csv", "source_hashes_status.json")}
    lines = [f"# {status['cell']} 冻结 OOF 诊断", "", f"范围：完整 **{n} 位供体**，{len(tables['pooled_metrics'])} 个 protocol/model 组合；只读取冻结预测，不训练、不调参、不改年龄校准，不把旧 977 人 OOF 截成 505 人模型。",
             "", "## 1. 总体指标与五折汇总", "", "下列为 pooled OOF MAE / RMSE / R²；括号为五外折**不加权均值 ± 样本 SD（ddof=1）**。", "",
             "|协议/模型|MAE 年|RMSE 年|R²|", "|---|---:|---:|---:|"]
    for row in tables["fold_summary"]:
        values = [f"{row['pooled_' + metric]:.4f} ({row['unweighted_fold_mean_' + metric]:.4f} ± {row['fold_sample_sd_ddof1_' + metric]:.4f})" for metric in METRIC_NAMES]
        lines.append(f"|{row['protocol']}/{row['model']}|" + "|".join(values) + "|")
    lines += ["", "折样本数加权 MAE 与 pooled MAE 均在数值容差内相等。RMSE 需先按 n 加权折 MSE 再开方；pooled R² 用全体年龄均值作分母，**不是折 R² 的平均值**。SD 不是置信区间；训练集重叠，五折不独立。",
              "", "## 2. 性别分层与年龄背景", "", "|协议/模型|性别|n|年龄均值|年龄方差 ddof=1|MAE 年|RMSE 年|R²|", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in tables["sex_stratified_metrics"]:
        lines.append(f"|{row['protocol']}/{row['model']}|{row['sex']}|{row['n']}|{row['age_mean']:.2f}|{row['age_variance_ddof1']:.2f}|{row['mae_years']:.4f}|{row['rmse_years']:.4f}|{row['r2']:.4f}|")
    lines += ["", "分层 MAE/RMSE 受年龄构成影响；分层 R² 还受本层年龄方差影响，不能直接当作性别公平性或生物差异的证明。", "", "## 3. 细胞数与实际文库 UMI 的误差关联", "",
              "signed residual = 预测年龄 − 实际年龄；正值为高估。表内 `raw → adjusted`：原始 Spearman → 调整后秩残差相关。两种 exposure 分别分析，**没有互相调整**。", "",
              "|协议/模型|exposure|绝对 OOF 误差 raw → adjusted|有符号残差 raw → adjusted|n|", "|---|---|---:|---:|---:|"]
    for base in tables["pooled_metrics"]:
        for exposure in cell_fields(status["cell"]):
            rows = [r for r in tables["error_depth_associations"] if r["protocol"] == base["protocol"] and r["model"] == base["model"] and r["exposure"] == exposure]
            values = [f"{r['raw_spearman']:+.4f} → {r['adjusted_rank_residual_r']:+.4f}" for r in rows]
            lines.append(f"|{base['protocol']}/{base['model']}|{exposure}|{values[0]}|{values[1]}|{base['n']}|")
    lines += ["", "调整方法：exposure 与误差各自取平均秩，分别对同一设计矩阵作 OLS 投影；设计含截距、年龄三次 B-spline（5 个年龄分位数结点）、sex 与 pool 固定效应，再取两组秩残差的 Pearson 相关。SVD 处理共线性；完整样本数、设计秩和残差自由度见 CSV。该数值是**调整后秩残差诊断，不是普通 Spearman，也不是因果效应**，不提供容易被误读的 iid p 值。",
              "", "年龄两端的误差可能呈 U 型，单一线性年龄调整不足；这里用非线性年龄样条缓解，但不能证明已充分控制年龄/批次/细胞组成，也没有对模型重新校准。UMI 是所选细胞 lineage 的实际 library 总 UMI（不是源表 source UMI、不是每细胞深度或测序 reads）；总 UMI 与细胞数相关，不能从该诊断分离技术深度的独立作用。正负关联均按观测结果报告，不能预设深度越大误差越小。",
              "", "## 4. 分带、验证及限制", "", f"- 固定细胞数分带：30–59、60–119、120–239、≥240；若出现 <30 则单列。深度按完整共同队列的实际 UMI Q1–Q4 分位边界分带，同深度不拆开；各带给 n、年龄方差、sex 构成和误差，见 {links['cell_depth_band_metrics.csv']}。",
              f"- 18–39、40–59、60–79、80–100 岁误差背景见 {links['age_context_metrics.csv']}；这是描述性补充，不评估新模型。分带 R² 可因年龄方差很窄而明显为负，不应与总体 R² 直接比较。",
              "- **min30 仅为既有纳入阈值，未被该分析验证为足够、最佳或无偏阈值**；分带关联不能验证 min30，也不能替代下采样/独立验证。",
              f"- 从 OOF 重算并核对原始 pooled/fold 指标：{status['source_metric_checks']} 个数值核验全部通过；atol=rtol={ATOL:g}。权重恒等式、完整供体覆盖、每组五折、元数据年龄/池一致性、重复预测与 pool_cv 池隔离均通过。",
              f"- 源文件 SHA-256 及运行状态见 {links['source_hashes_status.json']}；运行前后校验输入哈希不变。新增 OOF 必须完整覆盖同一队列，且 protocol/model/donor 键不重复、折号匹配；补充组没有旧报告指标时只注明未有原始参照，不冒称验证。",
              "- 输出仅聚合 CSV/图与状态；不发布 donor 行。所有切片和关联均探索性，多重比较、共享训练折、pool 结构及潜在未测量混杂限制解释；不是外部验证或新 505 人模型。", ""]
    en_bands = [r for r in tables["cell_depth_band_metrics"] if r["protocol"] == "donor_cv" and r["model"] == "elastic_net"]
    if en_bands:
        cells = [r for r in en_bands if r["stratifier"] == "cell_count"]
        depths = [r for r in en_bands if r["stratifier"] == "actual_umi_quartile"]
        en_ages = [r for r in tables["age_context_metrics"] if r["protocol"] == "donor_cv" and r["model"] == "elastic_net"]
        lines += ["### Donor-CV Elastic Net 的分带背景", "",
                  "细胞数各带 n / MAE 年：" + "；".join(f"{r['band']}: {r['n']} / {r['mae_years']:.2f}" for r in cells) + "。",
                  "实际 UMI Q1→Q4 的 MAE 年：" + " / ".join(f"{r['mae_years']:.2f}" for r in depths) + "；这些是未调整描述，不能视为深度效应。",
                  "年龄段 n / MAE 年：" + "；".join(f"{r['age_band']}: {r['n']} / {r['mae_years']:.2f}" for r in en_ages) + "。年龄构成与非线性误差关系均限制深度关联解释。", ""]
    if status["supplement_groups_without_original_metrics"]:
        lines += ["补充预测无原始指标可核对：" + ", ".join(status["supplement_groups_without_original_metrics"]) + "。", ""]
    return "\n".join(lines)


def run(args):
    output = prepare_output(args.output)
    cell = args.cell
    data = resolve_dir(args.data, cell, "donor_metadata.csv")
    models = resolve_dir(args.models, cell, "oof_predictions.csv")
    training = resolve_dir(args.training, cell, "metrics.csv", training=True)
    extra_oof = [Path(p).resolve() for p in args.extra_oof]
    cohort_path, summary_path = models / "analysis_cohort.csv", training / "summary.json"
    additional_paths = [p for p in (cohort_path, summary_path) if p.is_file()]
    paths = [data / "donor_metadata.csv", models / "fold_assignments.csv", models / "oof_predictions.csv",
             training / "metrics.csv", training / "fold_metrics.csv"] + extra_oof + additional_paths
    if any(output == p or output in p.parents for p in paths):
        raise ValueError("Output directory cannot contain source inputs")
    hashes = {str(p): sha256(p) for p in paths}
    cohort = read_csv(cohort_path) if cohort_path.is_file() else None
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else None
    metadata = select_analysis_metadata(read_csv(paths[0]), cohort, summary, cell)
    groups, base_keys = validate_and_join([read_csv(paths[2])] + [read_csv(p) for p in extra_oof],
                                         metadata, read_csv(paths[1]), cell=cell)
    tables, cuts = compute_tables(groups, cell=cell)
    comparisons = compare_source_metrics(tables["pooled_metrics"], tables["fold_metrics_recomputed"],
                                          read_csv(paths[3]), read_csv(paths[4]), base_keys)
    if not all(r["passed"] for r in comparisons):
        raise ValueError("Source metric numerical verification failed")
    if not all(r["weighted_mae_matches_pooled"] for r in tables["fold_summary"]):
        raise ValueError("Fold-size-weighted MAE identity failed")
    if hashes != {str(p): sha256(p) for p in paths}:
        raise ValueError("Source files changed during audit")
    original_keys = {(r["protocol"], r["model"]) for r in read_csv(paths[3])}
    status = {"status": "passed", "cell": cell, "cohort_n": tables["pooled_metrics"][0]["n"],
              "protocol_model_groups": len(groups), "source_metric_checks": len(comparisons),
              "source_files_sha256": hashes, "sources_unchanged": True,
              "absolute_tolerance": ATOL, "relative_tolerance": RTOL,
              "depth_quantile_cutpoints_" + cell_fields(cell)[1]: cuts.tolist(),
              "analysis_cohort_source": str(cohort_path) if cohort is not None else "complete_included_preparation_metadata",
              "analysis_cohort_metadata_exact_match": True,
              "training_summary_n_donors_verified": summary is not None,
              "supplement_groups_without_original_metrics": [f"{p}/{m}" for p, m in sorted(set(groups) - original_keys)],
              "all_weighted_mae_match_pooled": True, "participant_rows_written": False,
              "min30_validated": False, "causal_interpretation_supported": False,
              "adjustment": "separate_average_ranks_projection_on_age_bspline_degree3_5quantile_knots_sex_pool_then_residual_Pearson",
              "age_calibration_changed": False, "models_trained": False}
    # Recheck immediately before creating/writing output, preserving historical runs.
    prepare_output(output)
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        write_csv(output / (name + ".csv"), rows)
    write_csv(output / "source_metric_verification.csv", comparisons)
    make_plots(tables, output)
    if hashes != {str(p): sha256(p) for p in paths}:
        raise ValueError("Source files changed during plot/report creation")
    (output / "REPORT.md").write_text(report(tables, status, output), encoding="utf-8")
    (output / "source_hashes_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: status[k] for k in ("status", "cohort_n", "protocol_model_groups", "source_metric_checks", "sources_unchanged")}, ensure_ascii=False))
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cell", default="CD8")
    parser.add_argument("--data", type=Path, default=ROOT / "data/processed/CD8")
    parser.add_argument("--models", type=Path, default=ROOT / "data/models/CD8")
    parser.add_argument("--training", type=Path, default=ROOT / "results/cd8_training")
    parser.add_argument("--output", type=Path, default=ROOT / "results/methodology_audit/cd8_oof")
    parser.add_argument("--extra-oof", type=Path, action="append", default=[], help="Additional disjoint full-cohort OOF CSV; repeatable")
    args = parser.parse_args()
    try:
        run(args)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        print(f"OOF audit failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
