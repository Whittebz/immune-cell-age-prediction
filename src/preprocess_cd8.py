#!/usr/bin/env python3
"""Stream curated OneK1K CD8 counts into donor-level pseudobulk; no modeling.

Expression outputs are deliberately participant-level and belong in gitignored data/.
The original H5AD is opened read-only. No cohort-wide gene selection is performed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import sys
import time
from datetime import datetime, timezone

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SUBTYPES = ("CD8 Naive", "CD8 TCM", "CD8 TEM", "CD8 Proliferating")
SOURCE_URL = "https://datasets.cellxgene.cziscience.com/1e44db10-b572-46cc-adae-dcc7acd44ca6.h5ad"
NORMALIZATION_NAME = "natural log1p(count / donor_actual_UMI * 1e6)"


def read_column(group: h5py.Group, name: str) -> np.ndarray:
    """Decode an AnnData column, rejecting missing categorical codes."""
    obj = group[name]
    if isinstance(obj, h5py.Dataset):
        return obj.asstr()[:] if obj.dtype.kind in "OSU" else obj[:]
    if "codes" not in obj or "categories" not in obj:
        raise ValueError(f"Unsupported column encoding: {obj.name}")
    codes = obj["codes"][:]
    cat = obj["categories"]
    categories = cat.asstr()[:] if cat.dtype.kind in "OSU" else cat[:]
    if np.any(codes < 0) or np.any(codes >= len(categories)):
        raise ValueError(f"Missing/invalid categorical codes in {obj.name}")
    return categories[codes]


def describe(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return {
        "n": int(values.size), "missing": int(values.size - finite.size),
        "min": float(np.min(finite)) if finite.size else None,
        "median": float(np.median(finite)) if finite.size else None,
        "max": float(np.max(finite)) if finite.size else None,
        "p05": float(np.percentile(finite, 5)) if finite.size else None,
        "p95": float(np.percentile(finite, 95)) if finite.size else None,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalise_counts(counts: np.ndarray, target_sum: float = 1e6) -> np.ndarray:
    """Independent-per-donor CPM then natural log1p; no fit across donors."""
    if not np.isfinite(target_sum) or target_sum <= 0:
        raise ValueError("target_sum must be finite and positive")
    if counts.ndim != 2 or np.any(counts < 0):
        raise ValueError("Expected a nonnegative two-dimensional count matrix")
    library = counts.sum(axis=1, dtype=np.int64)
    if np.any(library <= 0):
        raise ValueError("Cannot normalize an empty donor library")
    # Work rowwise in float64 for validation, save float32 for compact storage.
    result = np.empty(counts.shape, dtype=np.float32)
    for row, total in enumerate(library):
        cpm = counts[row].astype(np.float64) * (target_sum / float(total))
        if not np.isclose(cpm.sum(), target_sum, rtol=1e-10):
            raise ValueError("Pre-log CPM sum failed validation")
        result[row] = np.log1p(cpm)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite normalized expression")
    return result


def stream_counts(
    x: h5py.Group, cell_donor: np.ndarray, cell_subtype: np.ndarray,
    selected: np.ndarray, retained_donors: np.ndarray,
    source_umi: np.ndarray, source_features: np.ndarray, block_rows: int,
    subtypes=SUBTYPES, cell_key="CD8",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Read CSR row chunks, never materializing the cell-by-gene matrix."""
    n_cells, n_genes = map(int, x.attrs["shape"])
    n_donors = retained_donors.size
    indptr = x["indptr"][:]
    if (len(indptr) != n_cells + 1 or indptr[0] != 0
            or np.any(np.diff(indptr) < 0)
            or indptr[-1] != len(x["data"]) or len(x["indices"]) != len(x["data"])):
        raise ValueError("Invalid CSR structure")
    mapping = np.full(n_donors, -1, dtype=np.int64)
    mapping[retained_donors] = np.arange(int(retained_donors.sum()))
    counts = np.zeros((int(retained_donors.sum()), n_genes), dtype=np.int64)
    library_all = np.zeros(n_donors, dtype=np.int64)
    subtype_umi = np.zeros((n_donors, len(subtypes)), dtype=np.int64)
    umi_deltas, feature_deltas, umi_relative = [], [], []
    processed_cells = 0
    start_time = time.monotonic()
    for start in range(0, n_cells, block_rows):
        stop = min(start + block_rows, n_cells)
        local_selected = selected[start:stop]
        if not local_selected.any():
            continue
        lo, hi = int(indptr[start]), int(indptr[stop])
        values = x["data"][lo:hi]
        genes = x["indices"][lo:hi]
        if np.any(genes < 0) or np.any(genes >= n_genes):
            raise ValueError("CSR gene index out of bounds")
        rows = np.repeat(np.arange(stop - start), np.diff(indptr[start:stop + 1]))
        keep = local_selected[rows]
        selected_values = values[keep]
        if (not np.isfinite(selected_values).all() or np.any(selected_values < 0)
                or np.any(selected_values != np.rint(selected_values))
                or np.any(selected_values > 2**31 - 1)):
            raise ValueError("Selected X entries are not finite nonnegative integer counts")
        selected_values = selected_values.astype(np.int64)
        selected_rows = rows[keep]
        selected_genes = genes[keep]
        # Actual X counts are authoritative; original QC fields may differ slightly.
        row_sums = np.bincount(selected_rows, weights=selected_values,
                               minlength=stop - start).astype(np.int64)
        row_features = np.bincount(selected_rows[selected_values > 0],
                                   minlength=stop - start)
        positions = np.flatnonzero(local_selected) + start
        actual = row_sums[local_selected]
        if np.any(actual <= 0):
            raise ValueError("Selected cell has an empty count library")
        source = source_umi[positions]
        umi_deltas.append(actual.astype(float) - source)
        feature_deltas.append(row_features[local_selected].astype(float) - source_features[positions])
        umi_relative.append(np.abs(actual - source) / np.maximum(source, 1))
        np.add.at(library_all, cell_donor[positions], actual)
        np.add.at(subtype_umi, (cell_donor[positions], cell_subtype[positions]), actual)
        donor_rows = mapping[cell_donor[start:stop][selected_rows]]
        retained = donor_rows >= 0
        np.add.at(counts, (donor_rows[retained], selected_genes[retained]), selected_values[retained])
        processed_cells += len(positions)
        if start // block_rows % 20 == 0 or stop == n_cells:
            print(f"Read {stop:,}/{n_cells:,} cells; aggregated {processed_cells:,} {cell_key} cells "
                  f"({time.monotonic() - start_time:.1f}s)", flush=True)
    if processed_cells != int(selected.sum()):
        raise ValueError("Selected cell conservation failed")
    if not np.array_equal(counts.sum(axis=1, dtype=np.int64), library_all[retained_donors]):
        raise ValueError("Donor count conservation failed")
    udelta, fdelta = np.concatenate(umi_deltas), np.concatenate(feature_deltas)
    diagnostics = {
        "selected_cells_scanned": processed_cells,
        "all_selected_counts_finite_nonnegative_integer": True,
        "umi_source_mismatch_cells": int(np.count_nonzero(udelta)),
        "umi_actual_minus_source": describe(udelta),
        "umi_max_relative_discrepancy": float(np.concatenate(umi_relative).max()),
        "feature_source_mismatch_cells": int(np.count_nonzero(fdelta)),
        "features_actual_minus_source": describe(fdelta),
        "count_conservation_passed": True,
        "metadata_discrepancy_note": "Source QC fields can differ from published X; do not force equality or infer a confirmed cause.",
    }
    return counts, library_all, subtype_umi, diagnostics


def write_csv(path: Path, fieldnames: list[str], rows) -> None:
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_qc(path: Path, ages_before, ages_after, donor_cells, subtypes, metrics,
            cell_key="CD8", subtype_names=None) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    bins = np.arange(15, 106, 5)
    axes[0, 0].hist(ages_before, bins=bins, histtype="step", color="#c77c41", label="Before donor filter")
    axes[0, 0].hist(ages_after, bins=bins, alpha=.7, color="#287c8e", label="Retained")
    axes[0, 0].set(xlabel="Chronological age (years)", ylabel="Donors", title="A. Donor age distribution")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].hist(donor_cells, bins=40, color="#287c8e")
    axes[0, 1].axvline(metrics["min_cells"], color="#c77c41", linestyle="--", label="Minimum cells")
    axes[0, 1].set(xlabel=f"QC-passing {cell_key} cells per donor", ylabel="Donors", title="B. Cell coverage")
    axes[0, 1].legend(fontsize=8)
    axes[0, 2].bar(subtype_names or ["Naive", "TCM", "TEM", "Prolif."], subtypes, color="#287c8e")
    if subtype_names is not None:
        axes[0, 2].tick_params(axis="x", labelrotation=45, labelsize=8)
    axes[0, 2].set(ylabel="Retained cells", title=f"C. {cell_key} subtype composition")
    for ax, key, label, title in [
        (axes[1, 0], "n_features", "Source detected genes per cell", "D. Source cell QC: features"),
        (axes[1, 1], "mt_percent", "Source mitochondrial counts (%)", "E. Source cell QC: mitochondria"),
        (axes[1, 2], "annotation_score", "Source annotation score", "F. Annotation confidence (unfiltered)"),
    ]:
        ax.hist(metrics[key], bins=45, color="#287c8e")
        ax.set(xlabel=label, ylabel=f"{cell_key} cells", title=title)
    fig.suptitle(f"OneK1K {cell_key} preprocessing | descriptive QC, not model results", fontsize=15)
    fig.savefig(path, dpi=160)
    svg = io.StringIO()
    fig.savefig(svg, format="svg")
    path.with_suffix(".svg").write_text(
        "\n".join(line.rstrip() for line in svg.getvalue().splitlines()) + "\n", encoding="utf-8"
    )
    plt.close(fig)


def preprocess(
    input_path: Path, output_dir: Path, report_dir: Path,
    min_cells: int = 30, min_genes: int = 200, max_mt: float = 20,
    block_rows: int = 4096, plots: bool = True, hash_input: bool = True,
    subtypes=SUBTYPES, cell_key="CD8",
) -> dict:
    if not cell_key.isascii() or not cell_key.isalnum():
        raise ValueError("cell_key must be an ASCII alphanumeric lineage key")
    if isinstance(subtypes, str) or not subtypes or len(set(subtypes)) != len(subtypes):
        raise ValueError("subtypes must be a nonempty unique label sequence")
    subtypes = tuple(subtypes)
    cell_slug = cell_key.lower()
    if min_cells < 1 or min_genes < 0 or block_rows < 1 or not 0 <= max_mt <= 100:
        raise ValueError("Invalid QC or chunk parameters")
    input_path, output_dir, report_dir = input_path.resolve(), output_dir.resolve(), report_dir.resolve()
    if output_dir == report_dir or output_dir in report_dir.parents or report_dir in output_dir.parents:
        raise ValueError("Expression and public-report directories must be separate")
    for directory in (output_dir, report_dir):
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f"Refusing to overwrite a nonempty output directory: {directory}")
    before_stat = input_path.stat()
    print(f"Input: {input_path} ({before_stat.st_size:,} bytes)", flush=True)
    input_hash = sha256(input_path) if hash_input else None
    started = time.monotonic()
    with h5py.File(input_path, "r") as f:
        x, obs, var = f["X"], f["obs"], f["var"]
        if x.attrs.get("encoding-type") != "csr_matrix":
            raise ValueError("This processor requires CSR-encoded X")
        n_cells, n_genes = map(int, x.attrs["shape"])
        donors = read_column(obs, "donor_id").astype(str)
        age = read_column(obs, "age").astype(float)
        sex = read_column(obs, "sex").astype(str)
        pool = read_column(obs, "pool_number").astype(str)
        labels = read_column(obs, "predicted.celltype.l2").astype(str)
        n_features = read_column(obs, "nFeature_RNA").astype(float)
        mt = read_column(obs, "percent.mt").astype(float)
        source_umi = read_column(obs, "nCount_RNA").astype(float)
        confidence = read_column(obs, "predicted.celltype.l2.score").astype(float)
        for a in (donors, age, sex, pool, labels, n_features, mt, source_umi, confidence):
            if len(a) != n_cells:
                raise ValueError("Cell metadata length mismatch")
        if not np.isfinite(age).all() or np.any(age < 0) or np.any(donors == ""):
            raise ValueError("Missing or invalid donor/age metadata")
        donor_names, first, donor_codes = np.unique(donors, return_index=True, return_inverse=True)
        for name, values in (("age", age), ("sex", sex), ("pool_number", pool)):
            if not np.array_equal(values, values[first][donor_codes]):
                raise ValueError(f"Donor has inconsistent {name}; resolve repeat samples explicitly")
        n_donors = len(donor_names)
        subtype = np.full(n_cells, -1, dtype=np.int8)
        for index, label in enumerate(subtypes):
            subtype[labels == label] = index
        cd8 = subtype >= 0
        cell_qc = (np.isfinite(n_features) & (n_features >= min_genes)
                   & np.isfinite(mt) & (mt >= 0) & (mt <= max_mt)
                   & np.isfinite(source_umi) & (source_umi > 0))
        selected = cd8 & cell_qc
        before_counts = np.bincount(donor_codes[cd8], minlength=n_donors)
        cell_counts = np.bincount(donor_codes[selected], minlength=n_donors)
        retained = cell_counts >= min_cells
        if not selected.any() or not retained.any():
            raise ValueError("No cells/donors pass the configured QC")
        subtype_cells = np.zeros((n_donors, len(subtypes)), dtype=np.int64)
        np.add.at(subtype_cells, (donor_codes[selected], subtype[selected]), 1)
        gene_ids = read_column(var, var.attrs.get("_index", "_index")).astype(str)
        gene_symbols = read_column(var, "feature_name").astype(str)
        if len(gene_ids) != n_genes or len(np.unique(gene_ids)) != n_genes:
            raise ValueError("Gene IDs must be unique and match the count matrix")
        print(f"{cell_key} cells: {cd8.sum():,}; cell-QC passing: {selected.sum():,}; "
              f"retained donors: {retained.sum():,}/{n_donors:,}", flush=True)
        counts, library_all, subtype_umi, diagnostics = stream_counts(
            x, donor_codes, subtype, selected, retained, source_umi, n_features, block_rows,
            subtypes=subtypes, cell_key=cell_key,
        )
    expression = normalise_counts(counts)
    after_stat = input_path.stat()
    if (before_stat.st_size, before_stat.st_mtime_ns) != (after_stat.st_size, after_stat.st_mtime_ns):
        raise ValueError("Input file changed during preprocessing")
    age_by_donor, sex_by_donor, pool_by_donor = age[first], sex[first], pool[first]
    included_cells = int(cell_counts[retained].sum())
    source_totals = np.bincount(donor_codes[selected], weights=source_umi[selected], minlength=n_donors)
    meta_rows = []
    for d, name in enumerate(donor_names):
        row = {
            "donor_id": name, "age": float(age_by_donor[d]), "sex": sex_by_donor[d],
            "pool_number": pool_by_donor[d], "included": int(retained[d]),
            "exclusion_reason": "" if retained[d] else f"fewer_than_min_{cell_slug}_cells",
            f"n_{cell_slug}_cells_before_qc": int(before_counts[d]), f"n_{cell_slug}_cells": int(cell_counts[d]),
            f"actual_{cell_slug}_umi": int(library_all[d]), f"source_{cell_slug}_umi": int(source_totals[d]),
        }
        for k, label in enumerate(subtypes):
            key = label.replace(" ", "_").lower()
            row[f"n_{key}"] = int(subtype_cells[d, k])
            row[f"fraction_{key}"] = float(subtype_cells[d, k] / cell_counts[d]) if cell_counts[d] else 0.0
            row[f"umi_{key}"] = int(subtype_umi[d, k])
        meta_rows.append(row)
    summary = {
        "status": "completed", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": {"path": str(input_path), "bytes": before_stat.st_size,
                  "sha256": input_hash, "source_url": SOURCE_URL,
                  "source_cells": n_cells, "source_genes": n_genes, "observed_donors": n_donors},
        "parameters": {"cell_key": cell_key, "bundle_name": f"{cell_slug}_pseudobulk.npz",
                       "subtype_column": "predicted.celltype.l2", "subtypes": list(subtypes),
                       "min_cells_per_donor": min_cells, "min_source_genes_per_cell": min_genes,
                       "max_source_mt_percent": max_mt, "block_rows": block_rows,
                       "annotation_score_filter": None, "normalization": NORMALIZATION_NAME},
        "cells": {f"{cell_slug}_before_qc": int(cd8.sum()), f"{cell_slug}_after_cell_qc": int(selected.sum()),
                  f"{cell_slug}_in_retained_donors": included_cells,
                  "removed_by_cell_qc": int((cd8 & ~cell_qc).sum()),
                  "removed_by_donor_filter": int(selected.sum()) - included_cells,
                  "retained_by_subtype": {label: int(subtype_cells[retained, i].sum()) for i, label in enumerate(subtypes)}},
        "donors": {"before": n_donors, "retained": int(retained.sum()),
                   "excluded": int((~retained).sum()),
                   "age_before": describe(age_by_donor), "age_retained": describe(age_by_donor[retained]),
                   "excluded_ages": sorted(age_by_donor[~retained].tolist()),
                   "retained_cell_count": describe(cell_counts[retained]),
                   "retained_sex_counts": {s: int(np.sum(sex_by_donor[retained] == s)) for s in np.unique(sex_by_donor)},
                   "retained_pool_count": int(len(np.unique(pool_by_donor[retained])))},
        "features": {"retained_genes": n_genes, "unique_gene_symbols": int(len(np.unique(gene_symbols))),
                     "nonzero_in_retained_donors": int(np.any(counts > 0, axis=0).sum()),
                     "gene_selection_performed": False, "primary_gene_key": "Ensembl ID"},
        "validation": {**diagnostics, "donor_metadata_consistent": True,
                       "raw_integer_count_sum": int(counts.sum(dtype=np.int64)),
                       "output_shape": list(counts.shape), "normalized_expression_finite": True,
                       "pre_log_cpm_target": 1e6},
        "source_cell_qc": {"n_features": describe(n_features[cd8]), "mt_percent": describe(mt[cd8]),
                           "annotation_score": describe(confidence[cd8])},
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "h5py": h5py.__version__},
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "limitations": [
            "Source is already curated; these guardrails are not de novo doublet/ambient-RNA removal.",
            f"Broad {cell_key} pseudobulk mixes subtype proportions and within-subtype expression; profiles are RNA-weighted.",
            "Thirty cells is a pragmatic threshold, not a guarantee of measurement precision; report age attrition.",
            "Pool is a shared technical batch; donor mapping does not establish absence of repeated biological sampling.",
            "CPM is relative expression; normalization sensitivity and training-fold feature selection remain future work.",
            "No model, benchmark, biological-age ground truth, or clinical claim is produced by this step.",
        ],
    }
    for directory in (output_dir, report_dir):
        directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / f"{cell_slug}_pseudobulk.npz", counts=counts, log1p_cpm=expression,
                        donor_ids=donor_names[retained].astype(str), gene_ids=gene_ids, gene_symbols=gene_symbols)
    write_csv(output_dir / "donor_metadata.csv", list(meta_rows[0]), (r for r in meta_rows if r["included"]))
    write_csv(output_dir / "donor_qc.csv", list(meta_rows[0]), meta_rows)
    write_csv(output_dir / "gene_metadata.csv", ["gene_id", "gene_symbol"],
              ({"gene_id": g, "gene_symbol": s} for g, s in zip(gene_ids, gene_symbols)))
    if plots:
        plot_qc(report_dir / "qc_overview.png", age_by_donor, age_by_donor[retained], cell_counts,
                subtype_cells[retained].sum(axis=0), {
                    "min_cells": min_cells, "n_features": n_features[cd8],
                    "mt_percent": mt[cd8], "annotation_score": confidence[cd8],
                }, cell_key=cell_key, subtype_names=subtypes)
    summary["artifacts"] = {}
    for path in sorted(output_dir.iterdir()):
        summary["artifacts"][path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    notes = (
        f"# {cell_key} 数据处理报告\n\n"
        f"- 原始数据：{n_cells:,} 个细胞，{n_genes:,} 个基因，{n_donors} 位供者。\n"
        f"- {cell_key} 注释细胞：{int(cd8.sum()):,}；基本细胞质控后：{int(selected.sum()):,}。\n"
        f"- 每人至少 {min_cells} 个 {cell_key} 细胞：保留 **{int(retained.sum())} 位供者、{included_cells:,} 个细胞**。\n"
        f"- 排除 {int((~retained).sum())} 位供者，其年龄为 {summary['donors']['excluded_ages']}；规则未按年龄调优。\n"
        f"- 建模输入形状：**{int(retained.sum())} × {n_genes}**（供者 × 基因）。\n"
        "- 原始计数按供者求和，按实际表达矩阵的供者计数总和计算 CPM，再取自然对数 log1p。\n"
        "- 基因主键为唯一 Ensembl ID；基因符号有重复，不合并、不删除。\n\n"
        "## 质量检查\n\n"
        "- 所有选中计数分块检查为有限、非负、整数；细胞遍历数量及供者计数守恒通过。\n"
        f"- {diagnostics['umi_source_mismatch_cells']:,} 个选中细胞的实际计数和与原 nCount_RNA 不同。\n"
        f"- 实际减原始 UMI 字段的范围：{diagnostics['umi_actual_minus_source']['min']} 至 {diagnostics['umi_actual_minus_source']['max']}。\n"
        "- 不强行修改计数匹配原字段；差异原因未确认，归一化使用当前 X 的实际计数。\n"
        "- 源数据已经整理；当前检查不是重新完成双细胞剔除或完整单细胞质控。\n\n"
        "## 使用与限制\n\n"
        f"供者级数据保存在本地 {output_dir}；汇总 JSON 不含逐供者表达，请勿公开逐供者数据。\n"
        f"{cell_key} 汇总混合了配置中的亚群，可能同时反映亚群组成和细胞内表达变化。\n"
        "pool_number 是技术混样批次，应保留供后续敏感性分析；不以供者 ID 或年龄作预测特征。\n"
        "本步骤未筛选高变/年龄相关基因，未标准化整个队列，未训练模型；这些学习步骤必须在训练折内完成。\n"
        "年龄是实际年龄标签，不是独立测量的真实免疫年龄。\n"
    )
    (report_dir / "REPORT.md").write_text(notes, encoding="utf-8")
    # Manifest is written last, so its presence marks a fully completed run.
    (report_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Completed. Expression: {output_dir}; report: {report_dir}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "dataset/onek1k.h5ad")
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/CD8")
    parser.add_argument("--report", type=Path, default=ROOT / "results/cd8")
    parser.add_argument("--min-cells", type=int, default=30)
    parser.add_argument("--min-genes", type=int, default=200)
    parser.add_argument("--max-mt", type=float, default=20)
    parser.add_argument("--block-rows", type=int, default=4096)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--skip-input-hash", action="store_true")
    args = parser.parse_args()
    preprocess(args.input, args.output, args.report, args.min_cells, args.min_genes,
               args.max_mt, args.block_rows, not args.no_plots, not args.skip_input_hash)


if __name__ == "__main__":
    main()
