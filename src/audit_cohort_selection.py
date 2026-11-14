#!/usr/bin/env python3
"""Aggregate-only, metadata-only audit of the frozen common-cohort selection.

Run: .venv/bin/python -B src/audit_cohort_selection.py
No expression reads, protocol/engine imports, training, inference tests, or donor exports.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import itertools
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/methodology_audit/cohort_selection"
CELL_COLUMNS = {"CD4": "n_cd4_cells", "B": "n_b_cells", "NK": "n_nk_cells",
                "MONO": "n_mono_cells", "CD8": "n_cd8_cells"}
AGE_BANDS = ((18, 40, "18–39"), (40, 60, "40–59"),
             (60, 80, "60–79"), (80, 101, "80–100"))
COHORT_LABELS = {"full": "Full source", "common": "Common", "excluded": "Excluded"}
COLORS = {"full": "#4c5968", "common": "#127e86", "excluded": "#c16d3c"}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_digest(value):
    """Match frozen cell_pipeline.canonical_digest without importing engines."""
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def load_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if len(reader.fieldnames or []) != len(set(reader.fieldnames or [])):
            raise ValueError("Duplicate metadata columns")
        return list(reader)


def prepare_output(path):
    """Refuse any pre-existing content, including hidden files; never overwrite."""
    if path.is_symlink() or (path.exists() and not path.is_dir()):
        raise ValueError("Output must be a real directory, not a file or symlink")
    if path.exists() and any(path.iterdir()):
        raise ValueError("Output directory is nonempty; refusing to overwrite")
    path.mkdir(parents=True, exist_ok=True)


def write_csv(path, records):
    if not records:
        raise ValueError("Refusing an empty table without a schema")
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def write_text(path, text):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text.rstrip() + "\n")


def validate_metadata(eligibility, common, min_cells):
    required = {"donor_id", "age", "sex", "pool_number", "common_eligible", *CELL_COLUMNS.values()}
    if not eligibility or not common:
        raise ValueError("Metadata cannot be empty")
    if any(not required.issubset(row) for row in eligibility + common):
        raise ValueError("Metadata schema is missing required fields")
    ids = [row["donor_id"] for row in eligibility]
    selected_ids = [row["donor_id"] for row in common]
    if any(not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("Source identifiers must be nonempty and unique")
    if len(set(selected_ids)) != len(selected_ids):
        raise ValueError("Common identifiers must be unique")
    age = np.array([float(row["age"]) for row in eligibility])
    if not np.isfinite(age).all() or np.any((age < 18) | (age > 100)):
        raise ValueError("Ages must be finite and in the audited 18–100 range")
    sex = np.array([row["sex"] for row in eligibility])
    pool = np.array([row["pool_number"] for row in eligibility])
    if any(not value for value in sex) or any(not value for value in pool):
        raise ValueError("Sex and pool values must be present")
    counts = np.array([[int(row[column]) for column in CELL_COLUMNS.values()]
                       for row in eligibility])
    if np.any(counts < 0):
        raise ValueError("Cell counts cannot be negative")
    if any(row["common_eligible"] not in {"0", "1"} for row in eligibility):
        raise ValueError("Eligibility flag must be binary")
    selected = np.array([row["common_eligible"] == "1" for row in eligibility])
    if not np.array_equal(selected, np.all(counts >= min_cells, axis=1)):
        raise ValueError("Frozen eligibility flags disagree with configured cell thresholds")
    source = {row["donor_id"]: row for row in eligibility}
    if set(selected_ids) != {ids[i] for i in np.flatnonzero(selected)}:
        raise ValueError("Common membership disagrees with eligibility flags")
    # Check complete shared metadata (not just matching row counts or IDs).
    if any(any(row[key] != source[row["donor_id"]][key] for key in required)
           for row in common):
        raise ValueError("Common metadata disagree with source metadata")
    return age, sex, pool, counts, selected


def describe(values):
    q25, median, q75 = np.percentile(values, [25, 50, 75], method="linear")
    return {"n": len(values), "age_min": float(np.min(values)),
            "age_mean": float(np.mean(values)), "age_sd": float(np.std(values, ddof=1)),
            "age_median": float(median), "age_q25": float(q25), "age_q75": float(q75),
            "age_iqr": float(q75 - q25), "age_max": float(np.max(values))}


def standardized_difference(common, excluded, binary=False):
    """Signed common minus excluded SMD; no hypothesis test or sampling assumption."""
    mean_a, mean_b = float(np.mean(common)), float(np.mean(excluded))
    if binary:
        variance = (mean_a * (1 - mean_a) + mean_b * (1 - mean_b)) / 2
    else:
        variance = (float(np.var(common, ddof=1)) + float(np.var(excluded, ddof=1))) / 2
    difference = mean_a - mean_b
    smd = difference / np.sqrt(variance) if variance > 0 else (0.0 if difference == 0 else None)
    return {"common_mean_or_proportion": mean_a, "excluded_mean_or_proportion": mean_b,
            "common_minus_excluded": difference, "smd": None if smd is None else float(smd)}


def retention_rows(age, sex, pool, selected):
    rows = []
    groups = [("age_band", label, (age >= low) & (age < high))
              for low, high, label in AGE_BANDS]
    groups += [("sex", value, sex == value) for value in sorted(set(sex))]
    groups += [("pool", value, pool == value) for value in sorted(set(pool), key=lambda x: (len(x), x))]
    for dimension, group, mask in groups:
        n, kept = int(mask.sum()), int((mask & selected).sum())
        rows.append({"dimension": dimension, "group": str(group), "n_full": n,
                     "n_common": kept, "n_excluded": n - kept,
                     "retention_proportion": kept / n if n else None,
                     "retention_percent": 100 * kept / n if n else None})
    return rows


def failure_tables(counts, selected, min_cells):
    names = list(CELL_COLUMNS)
    failed = counts < min_cells
    cells = []
    for i, cell in enumerate(names):
        eligible = int((~failed[:, i]).sum())
        sole_failure = int((failed[:, i] & (failed.sum(axis=1) == 1)).sum())
        cells.append({"cell_type": cell, "minimum_cells": min_cells, "n_source": len(counts),
                      "n_eligible": eligible, "n_ineligible": len(counts) - eligible,
                      "eligible_percent_of_source": 100 * eligible / len(counts),
                      "n_common": int(selected.sum()),
                      "n_eligible_but_excluded": eligible - int(selected.sum()),
                      "n_excluded_solely_this_failure": sole_failure})
    patterns = []
    for bits in itertools.product((False, True), repeat=len(names)):
        mask = np.all(failed == bits, axis=1)
        n = int(mask.sum())
        patterns.append({"failed_cell_types": "+".join(name for name, bit in zip(names, bits) if bit) or "none",
                         "n_failed_types": sum(bits), "n_donors": n,
                         "percent_of_source": 100 * n / len(counts),
                         "percent_of_excluded": 100 * n / int((~selected).sum()) if any(bits) else None})
    overlaps = [{"failed_type_a": a, "failed_type_b": b,
                 "n_fail_both": int((failed[:, i] & failed[:, j]).sum())}
                for i, a in enumerate(names) for j, b in enumerate(names)]
    return cells, patterns, overlaps


def save_figure(fig, path):
    # Exclusive creation prevents accidental replacement even after initial output check.
    with path.with_suffix(".png").open("xb") as stream:
        fig.savefig(stream, format="png", dpi=180)
    buffer = io.StringIO()
    fig.savefig(buffer, format="svg", metadata={"Date": None})
    write_text(path.with_suffix(".svg"), "\n".join(line.rstrip() for line in buffer.getvalue().splitlines()))


def make_figures(output, demographics, sex_rows, histogram, retention):
    # Use the project's already-populated cache rather than creating external caches.
    os.environ["MPLCONFIGDIR"] = str(ROOT / ".cache/matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.hashsalt": "cohort-selection-audit"})
    fig, ax = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
    for cohort in COHORT_LABELS:
        rows = [row for row in histogram if row["cohort"] == cohort]
        ax.stairs([row["within_cohort_percent"] for row in rows],
                  [row["age_low_inclusive"] for row in rows] + [rows[-1]["age_high_exclusive"]],
                  label=f"{COHORT_LABELS[cohort]} (n={demographics[cohort]['n']})",
                  color=COLORS[cohort], linewidth=2)
    ax.set(xlabel="Age (years; fixed 5-year bins)", ylabel="Within-cohort donors (%)",
           title="Age distribution: normalized separately within each cohort", xlim=(15, 105))
    ax.legend(frameon=False)
    save_figure(fig, output / "age_distribution")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.8, 4.8), constrained_layout=True)
    cohorts = list(COHORT_LABELS)
    bottom = np.zeros(len(cohorts))
    for index, category in enumerate(sorted({row["sex"] for row in sex_rows})):
        values = [next(row["percent"] for row in sex_rows if row["cohort"] == cohort and row["sex"] == category)
                  for cohort in cohorts]
        ax.bar(cohorts, values, bottom=bottom, label=category,
               color=["#40799b", "#cf8655", "#8b84a2"][index % 3], width=.58)
        for x, value, baseline in zip(range(len(cohorts)), values, bottom):
            ax.text(x, baseline + value / 2, f"{value:.1f}%", ha="center", va="center", color="white")
        bottom += values
    ax.set_xticks(range(len(cohorts)), [f"{COHORT_LABELS[c]}\nn={demographics[c]['n']}" for c in cohorts])
    ax.set(ylabel="Within-cohort donors (%)", ylim=(0, 100), title="Recorded sex proportions")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1, 1))
    save_figure(fig, output / "sex_proportions")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.8, 4.8), constrained_layout=True)
    rows = [row for row in retention if row["dimension"] == "age_band"]
    heights = [row["retention_percent"] if row["retention_percent"] is not None else 0 for row in rows]
    ax.bar(range(len(rows)), heights, color=COLORS["common"], width=.62)
    for x, row, height in zip(range(len(rows)), rows, heights):
        label = f"{row['n_common']}/{row['n_full']}\n{height:.1f}%" if row["n_full"] else "0/0; undefined"
        ax.text(x, height + 2, label, ha="center", va="bottom")
    ax.axhline(100 * demographics["common"]["n"] / demographics["full"]["n"],
               color=COLORS["full"], linestyle="--", label="Overall source retention")
    ax.set_xticks(range(len(rows)), [row["group"].replace("–", "-") for row in rows])
    ax.set(xlabel="Fixed age band (years)", ylabel="Common / source within band (%)",
           ylim=(0, 110), title="Common-cohort retention by source age band")
    ax.legend(frameon=False)
    save_figure(fig, output / "retention_by_age")
    plt.close(fig)
    return matplotlib.__version__


def report_text(summary):
    d = summary["demographics"]
    retention = summary["retention"]
    cells = {row["cell_type"]: row for row in summary["cell_eligibility"]}
    patterns = summary["failure_patterns"]
    age_smd = next(row for row in summary["selection_smd"] if row["variable"] == "age")
    younger = d["excluded"]["age_mean"] < d["common"]["age_mean"]
    lines = ["# 共同队列选择审计（仅元数据）", "", "## 方法与分母", "",
             "本审计只读取冻结的 eligibility、common_cohort、cell_types 配置及 protocol 元数据；不读取表达矩阵，不训练模型，不改动配置、引擎或主结果。",
             f"宏观源队列为 {d['full']['n']} 人，共同队列 {d['common']['n']} 人，排除 {d['excluded']['n']} 人；总体保留率 {100*d['common']['n']/d['full']['n']:.2f}%。",
             f"共同资格要求五类细胞每类均 ≥{summary['minimum_cells']} 个。CD8 单类合格 {cells['CD8']['n_eligible']} 人，不能把该数当作宏观源队列分母；MONO 单类合格 {cells['MONO']['n_eligible']} 人，是主要数量瓶颈。",
             "已核对供者唯一性、共同队列成员及所有共享元数据、五类阈值与 common_eligible 标志；供者标识仅在内存中用于核验，不写入审计产物。年龄无缺失且在 18–100 岁，sex/pool 无缺失，细胞计数非负。",
             "年龄 SD 使用样本标准差（ddof=1）；四分位数用线性插值，IQR=Q75−Q25。固定年龄层为 [18,40)、[40,60)、[60,80)、[80,101)，标签为18–39/40–59/60–79/80–100。sex 按源记录原值统计，不推断性别身份。",
             "年龄图为固定 5 岁箱、各队列独立归一化的百分比分布（每队列合计100%），不是人数直接叠加。性别图同样使用队列内比例。保留率为同一分层内 common/source。", "", "## 结果", "",
             "|队列|n|年龄 min–max|均值±SD|中位数 [Q25,Q75]|IQR|", "|---|---:|---:|---:|---:|---:|"]
    for cohort, label in (("full", "全体"), ("common", "共同"), ("excluded", "排除")):
        r = d[cohort]
        lines.append(f"|{label}|{r['n']}|{r['age_min']:.0f}–{r['age_max']:.0f}|{r['age_mean']:.2f}±{r['age_sd']:.2f}|{r['age_median']:.1f} [{r['age_q25']:.1f}, {r['age_q75']:.1f}]|{r['age_iqr']:.1f}|")
    lines += ["", f"排除者平均年龄实际{'更低' if younger else '不低于共同队列'}；不能预设排除者更老。共同减排除的年龄均值差为 {age_smd['common_minus_excluded']:.2f} 岁，描述性 SMD={age_smd['smd']:.3f}。", "", "### sex 分布与比例", "",
              "|队列|sex|n|比例|", "|---|---|---:|---:|"]
    for row in summary["sex_distribution"]:
        lines.append(f"|{row['cohort']}|{row['sex']}|{row['n']}|{row['percent']:.2f}%|")
    lines += ["", "### 固定年龄层和 sex 的保留率", "", "|分层|组|共同/源|保留率|", "|---|---|---:|---:|"]
    for row in retention:
        if row["dimension"] != "pool":
            pct = f"{row['retention_percent']:.2f}%" if row["n_full"] else "未定义"
            lines.append(f"|{row['dimension']}|{row['group']}|{row['n_common']}/{row['n_full']}|{pct}|")
    pools = [row for row in retention if row["dimension"] == "pool" and row["n_full"]]
    low, high = min(pools, key=lambda r: r["retention_percent"]), max(pools, key=lambda r: r["retention_percent"])
    lines += ["", f"按 pool 保留率范围为 {low['retention_percent']:.2f}%（pool {low['group']}，{low['n_common']}/{low['n_full']}）至 {high['retention_percent']:.2f}%（pool {high['group']}，{high['n_common']}/{high['n_full']}）；完整 pool 汇总见 retention.csv。", "", "### 细胞阈值与重叠排除原因", "",
              "|细胞类|合格|不合格|仅此一类不合格|", "|---|---:|---:|---:|"]
    for row in summary["cell_eligibility"]:
        lines.append(f"|{row['cell_type']}|{row['n_eligible']}|{row['n_ineligible']}|{row['n_excluded_solely_this_failure']}|")
    mono_only = cells["MONO"]["n_excluded_solely_this_failure"]
    mono_fail = cells["MONO"]["n_ineligible"]
    lines += ["", f"MONO 不足涉及 {mono_fail}/{d['excluded']['n']} 名排除者（{100*mono_fail/d['excluded']['n']:.2f}%）；其中仅 MONO 不足 {mono_only} 人。MONO 合格但最终被其他门槛排除 {cells['MONO']['n_eligible_but_excluded']} 人。多原因重叠不可将各类不合格人数直接相加。", "", "实际非空排除组合："]
    for row in sorted(patterns, key=lambda r: -r["n_donors"]):
        if row["n_failed_types"] and row["n_donors"]:
            lines.append(f"- {row['failed_cell_types']}：{row['n_donors']} 人。")
    lines += ["", "## 解释与局限", "",
              "- full 与 common 重叠，因此不进行独立双样本 t 检验、KS 检验或其他独立样本 p 值检验。SMD 仅比较不重叠的 common 与 excluded，用作有限样本描述，不作随机抽样推断。",
              "- 连续年龄 SMD=(mean_common−mean_excluded)/sqrt((SD_common²+SD_excluded²)/2)，SD 为 ddof=1；二元 sex 或 pool 水平 SMD=(p_common−p_excluded)/sqrt((p_common(1−p_common)+p_excluded(1−p_excluded))/2)。正号表示 common 更高；零方差且均值相同记0，不同记 null。多水平分类以每一水平相对其余水平报告，不将其视为独立检验。",
              "- 年龄、sex 与 pool 保留率差异提供与已测变量相关的非随机选择线索，而不是机制证明或因果效应。并未校正 pool、年龄、sex 间的混杂，也未对 pool 聚类做推断。不能把相近的人口学分布解释为随机选择，更不能据此排除未测免疫健康相关的 MNAR。",
              "- 细胞回收/组成、技术批次、未测健康状态可能影响达标与否。本审计只核验冻结元数据计数阈值，不重新验证表达矩阵或技术 QC；结论仅适用于本源队列，不证明对其他人群可推广。",
              "- 本报告及全部 CSV/JSON/PNG/SVG 均为汇总结果，无供者ID，无个体年龄/性别/细胞计数逐行导出。", "", "## 来源、复现与产物", "",
              "运行命令：`.venv/bin/python -B src/audit_cohort_selection.py`。默认输出目录非空时拒绝运行，不覆盖已有审计或主分析产物。",
              "源文件字节 SHA-256、配置声明的原始 h5ad SHA-256（未读取或重算 h5ad）、冻结 protocol 一致性检查、Python/依赖版本及审计脚本哈希均见 summary.json。protocol 的 config_sha256 是 JSON 规范化后摘要（UTF-8、ensure_ascii=False、sort_keys=True、separators=(',', ':')），与原始文件字节哈希分开记录和核对；不混用两种哈希。",
              "`cohort_demographics.csv`、`sex_distribution.csv`、`age_distribution.csv`、`retention.csv`、`cell_eligibility.csv`、`failure_patterns.csv`（含全部32种组合，互斥并覆盖981人）、`cell_failure_overlap.csv`（对角线为各类不足人数，非对角线为两类同时不足，非互斥）、`selection_smd.csv`。三张图各提供 PNG/SVG：age_distribution、sex_proportions、retention_by_age。CSV 使用 LF，SVG 去除行尾空白。"]
    return "\n".join(lines)


def run(output=OUTPUT):
    if output.resolve() != OUTPUT.resolve() or output.is_symlink():
        raise ValueError("This scoped audit may only write its assigned output directory")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory is nonempty; refusing to overwrite")
    paths = {"eligibility": ROOT / "data/shared/eligibility.csv",
             "common_cohort": ROOT / "data/shared/common_cohort.csv",
             "cell_types_config": ROOT / "config/cell_types.json",
             "frozen_protocol": ROOT / "data/shared/protocol.json"}
    config = json.loads(paths["cell_types_config"].read_text(encoding="utf-8"))
    protocol = json.loads(paths["frozen_protocol"].read_text(encoding="utf-8"))
    min_cells = config["qc"]["min_cells"]
    if not isinstance(min_cells, int) or min_cells != 30 or set(config["cells"]) != set(CELL_COLUMNS):
        raise ValueError("This scoped audit requires the expected minimum of 30 and five cell classes")
    eligibility, common = load_csv(paths["eligibility"]), load_csv(paths["common_cohort"])
    age, sex, pool, counts, selected = validate_metadata(eligibility, common, min_cells)
    if (len(age), int(selected.sum()), int((~selected).sum())) != (981, 505, 476):
        raise ValueError("Source sizes differ from this scoped 981/505/476 audit")
    hashes = {name: sha256(path) for name, path in paths.items()}
    checks = {"source_hash_declarations_match": config["source_sha256"] == protocol["source_sha256"],
              "canonical_config_hash_matches_frozen_protocol": canonical_digest(config) == protocol["config_sha256"],
              "cohort_hash_matches_frozen_protocol": hashes["common_cohort"] == protocol["cohort_sha256"],
              "common_size_matches_protocol": len(common) == protocol["n_common_donors"],
              "membership_and_shared_metadata_match": True, "threshold_flags_match": True}
    if not all(checks.values()):
        raise ValueError("Source provenance disagrees with frozen protocol")
    masks = {"full": np.ones(len(age), dtype=bool), "common": selected, "excluded": ~selected}
    demographics = {name: describe(age[mask]) for name, mask in masks.items()}
    sex_rows = [{"cohort": name, "sex": str(category), "n": int(((sex == category) & mask).sum()),
                 "proportion": float(((sex == category) & mask).sum() / mask.sum()),
                 "percent": float(100 * ((sex == category) & mask).sum() / mask.sum())}
                for name, mask in masks.items() for category in sorted(set(sex))]
    edges = np.arange(15, 106, 5)
    histogram = []
    for name, mask in masks.items():
        frequencies, _ = np.histogram(age[mask], bins=edges)
        histogram.extend({"cohort": name, "age_low_inclusive": int(lo), "age_high_exclusive": int(hi),
                          "n": int(n), "within_cohort_percent": 100 * int(n) / int(mask.sum())}
                         for lo, hi, n in zip(edges[:-1], edges[1:], frequencies))
    retention = retention_rows(age, sex, pool, selected)
    cells, patterns, overlaps = failure_tables(counts, selected, min_cells)
    smds = [{"variable": "age", "level": "years", "type": "continuous",
             **standardized_difference(age[selected], age[~selected])}]
    for variable, values in (("sex", sex), ("pool", pool)):
        for level in sorted(set(values)):
            indicator = (values == level).astype(float)
            smds.append({"variable": variable, "level": str(level), "type": "binary_indicator",
                         **standardized_difference(indicator[selected], indicator[~selected], binary=True)})
    summary = {"audit": "cohort_selection_metadata_only", "created_at_utc": datetime.now(timezone.utc).isoformat(),
               "minimum_cells": min_cells, "source_files": {name: {"path": str(path.relative_to(ROOT)),
               "bytes": path.stat().st_size, "sha256": hashes[name]} for name, path in paths.items()},
               "declared_expression_source": {"sha256": config["source_sha256"],
               "bytes": config["source_bytes"], "read_or_rehashed": False},
               "audit_code": {"path": "src/audit_cohort_selection.py", "sha256": sha256(Path(__file__))},
               "config_canonical_digest": {"sha256": canonical_digest(config),
               "protocol_recorded_sha256": protocol["config_sha256"],
               "method": "SHA256 of UTF-8 JSON, ensure_ascii=False, sort_keys=True, separators=(',', ':'); separate from raw file SHA256"},
               "validation_checks": checks, "demographics": demographics,
               "sex_distribution": sex_rows, "retention": retention, "cell_eligibility": cells,
               "failure_patterns": patterns, "selection_smd": smds,
               "methods": {"age_sd_ddof": 1, "quantile_method": "linear", "age_histogram_bin_width_years": 5,
               "age_histogram_normalization": "within-cohort percent, sums to 100 for each cohort",
               "continuous_smd": "(mean_common-mean_excluded)/sqrt((sample_var_common+sample_var_excluded)/2)",
               "binary_smd": "(p_common-p_excluded)/sqrt((p_common*(1-p_common)+p_excluded*(1-p_excluded))/2)",
               "comparison": "common versus excluded (nonoverlapping), descriptive only; no p-values",
               "missing_data_policy": "fail on missing/nonfinite required metadata; no imputation",
               "expression_reads": False, "model_training": False, "donor_level_outputs": False}}
    prepare_output(output)
    tables = {"cohort_demographics": [{"cohort": name, **value} for name, value in demographics.items()],
              "sex_distribution": sex_rows, "age_distribution": histogram, "retention": retention,
              "cell_eligibility": cells, "failure_patterns": patterns,
              "cell_failure_overlap": overlaps, "selection_smd": smds}
    for name, rows in tables.items():
        write_csv(output / f"{name}.csv", rows)
    matplotlib_version = make_figures(output, demographics, sex_rows, histogram, retention)
    summary["runtime"] = {"python": platform.python_version(), "executable": sys.executable,
                          "numpy": np.__version__, "matplotlib": matplotlib_version}
    write_text(output / "REPORT.md", report_text(summary))
    summary["output_artifacts"] = [{"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
                                    "sha256": sha256(path)} for path in sorted(output.iterdir()) if path.is_file()]
    write_text(output / "summary.json", json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
    print(json.dumps({"output": str(output), "n_full": len(age), "n_common": int(selected.sum()),
                      "n_excluded": int((~selected).sum()), "age_smd_common_minus_excluded": smds[0]["smd"],
                      "artifacts": len(list(output.iterdir()))}, ensure_ascii=False))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    try:
        run()
    except (ValueError, OSError, KeyError) as exc:
        # Do not echo offending source values or donor IDs in validation errors.
        print(f"Audit refused or failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
