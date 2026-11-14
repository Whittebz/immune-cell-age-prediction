#!/usr/bin/env python3
"""Describe four post-hoc CD8 model features without retraining or causal claims."""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
from scipy.stats import spearmanr

from preprocess_cd8 import ROOT, sha256

CANDIDATES = (
    ("LRRN3", "ENSG00000173114", ["R1", "R7"], "Age-associated blood transcriptional marker"),
    ("CDKN2A", "ENSG00000147889", ["R2", "R3"], "Cell-cycle locus; gene-level RNA is not p16-specific"),
    ("CD70", "ENSG00000125726", ["R4", "R5"], "Activation/costimulation context, not a senescence diagnosis"),
    ("IGFBP3", "ENSG00000146674", ["R6"], "Growth-factor context; serum protein is not CD8 RNA"),
)
AGE_BINS = ((18, 40), (40, 60), (60, 80), (80, 101))


def rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_table(path, records):
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def bibtex_authors(display):
    shortened = display.replace(', et al.', '').rstrip('.')
    authors = []
    for author in shortened.split(', '):
        surname, initials = author.rsplit(' ', 1)
        formatted = ' '.join(f'{letter}.' for letter in initials.rstrip('.'))
        authors.append(f'{surname}, {formatted}')
    if 'et al.' in display:
        authors.append('others')
    return ' and '.join(authors)


def rank_correlation(x, y):
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(spearmanr(x, y).statistic)


def describe_gene(age, expression, counts, evidence):
    if len(age) != len(expression) or len(age) != len(counts):
        raise ValueError("Age, expression and counts must be aligned")
    if not np.isfinite(expression).all() or np.any(counts < 0):
        raise ValueError("Invalid gene measurements")
    coefficient = float(evidence["mean_signed_standardized_coefficient"])
    return {
        "gene_id": evidence["gene_id"], "gene_symbol": evidence["gene_symbol"],
        "mean_standardized_model_coefficient_years": coefficient,
        "model_direction": "higher_prediction" if coefficient > 0 else "lower_prediction",
        "available_folds": int(evidence["available_folds"]),
        "nonzero_folds": int(evidence["nonzero_folds"]),
        "sign_consistency": float(evidence["sign_consistency_when_nonzero"]),
        "same_cohort_age_spearman_descriptive": rank_correlation(age, expression),
        "n_donors": len(age), "n_donors_detected": int(np.sum(counts > 0)),
        "total_gene_umi": int(np.sum(counts)), "median_gene_umi": float(np.median(counts)),
        "median_log1p_cpm": float(np.median(expression)),
    }


def age_trends(age, expression, symbol):
    result = []
    for low, high in AGE_BINS:
        values = expression[(age >= low) & (age < high)]
        if len(values):
            result.append({
                "gene_symbol": symbol, "age_band": f"{low}-{high - 1}", "n_donors": len(values),
                "median_log1p_cpm": float(np.median(values)),
                "q25_log1p_cpm": float(np.percentile(values, 25)),
                "q75_log1p_cpm": float(np.percentile(values, 75)),
            })
    return result


def save_figure(fig, path):
    fig.savefig(path, dpi=170)
    buffer = io.StringIO()
    fig.savefig(buffer, format="svg")
    path.with_suffix(".svg").write_text(
        "\n".join(line.rstrip() for line in buffer.getvalue().splitlines()) + "\n", encoding="utf-8"
    )


def make_figures(output, selected, trends, age, naive, tem):
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    coefficient = [r["mean_standardized_model_coefficient_years"] for r in selected]
    ax.barh(np.arange(4), coefficient, color=["#287c8e" if v < 0 else "#bf7340" for v in coefficient])
    ax.set_yticks(np.arange(4), [f"{r['gene_symbol']} | {r['nonzero_folds']}/5 folds" for r in selected])
    ax.invert_yaxis()
    ax.axvline(0, color="#444444", linewidth=.8)
    ax.set(xlabel="Mean standardized model coefficient (years)", title="Four literature-interpretable CD8 model features")
    fig.suptitle("Model direction is not a causal protective/harmful effect", fontsize=11)
    save_figure(fig, output / "selected_gene_coefficients.png")
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for ax, gene in zip(axes.flat, selected):
        data = [r for r in trends if r["gene_symbol"] == gene["gene_symbol"]]
        x = np.arange(len(data))
        medians = np.array([r["median_log1p_cpm"] for r in data])
        color = "#287c8e" if gene["mean_standardized_model_coefficient_years"] < 0 else "#bf7340"
        ax.plot(x, medians, marker="o", color=color, linewidth=2)
        ax.fill_between(x, [r["q25_log1p_cpm"] for r in data], [r["q75_log1p_cpm"] for r in data], color=color, alpha=.18)
        ax.set_xticks(x, [f"{r['age_band']}\n(n={r['n_donors']})" for r in data])
        ax.set(xlabel="Chronological age band", ylabel="Donor pseudobulk log1pCPM",
               title=f"{gene['gene_symbol']} | descriptive Spearman {gene['same_cohort_age_spearman_descriptive']:.3f}")
    fig.suptitle("Same-cohort expression trends | line: median; shading: IQR, not a confidence interval", fontsize=12)
    save_figure(fig, output / "gene_age_trends.png")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, values, name in zip(axes, (naive, tem), ("Naive CD8 fraction", "Effector-memory CD8 fraction")):
        ax.scatter(age, values, s=11, alpha=.35, color="#287c8e", edgecolors="none")
        ax.set(xlabel="Chronological age (years)", ylabel="Fraction of QC-passing CD8 cells",
               title=f"{name}\nDescriptive Spearman {rank_correlation(age, values):.3f}")
    fig.suptitle("Subtype composition is an alternative explanation, not a tested causal mediator", fontsize=12)
    save_figure(fig, output / "composition_caveat.png")
    plt.close(fig)


def generate(data, training, references_path, output, plots=True):
    data, training, references_path, output = map(Path, (data, training, references_path, output))
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {output}")
    reference_doc = json.loads(references_path.read_text(encoding="utf-8"))
    references = {r["id"]: r for r in reference_doc["references"]}
    evidence_rows = rows(training / "elastic_net_feature_stability.csv")
    evidence = {r["gene_id"]: r for r in evidence_rows}
    metadata = rows(data / "donor_metadata.csv")
    with np.load(data / "cd8_pseudobulk.npz", allow_pickle=False) as bundle:
        ids, gene_ids, symbols = bundle["donor_ids"], bundle["gene_ids"], bundle["gene_symbols"]
        expression, counts = bundle["log1p_cpm"], bundle["counts"]
    if ids.tolist() != [r["donor_id"] for r in metadata]:
        raise ValueError("Donor row mismatch")
    age = np.array([float(r["age"]) for r in metadata])
    naive = np.array([float(r["fraction_cd8_naive"]) for r in metadata])
    tem = np.array([float(r["fraction_cd8_tem"]) for r in metadata])
    selected, trends = [], []
    for symbol, gene_id, reference_ids, context in CANDIDATES:
        index = np.flatnonzero(gene_ids == gene_id)
        if len(index) != 1 or symbols[index[0]] != symbol:
            raise ValueError(f"Ambiguous/mismatched gene annotation: {symbol}")
        info = evidence[gene_id]
        if (int(info["nonzero_folds"]) != 5 or float(info["sign_consistency_when_nonzero"]) != 1
                or info["stable_by_rule"] != "1"):
            raise ValueError(f"Prespecified explanatory feature no longer has expected stability: {symbol}")
        for reference in reference_ids:
            if reference not in references or symbol not in references[reference]["genes"]:
                raise ValueError("Reference registry mismatch")
        index = int(index[0])
        record = describe_gene(age, expression[:, index], counts[:, index], info)
        record["reference_ids"] = ";".join(reference_ids)
        record["interpretation_context"] = context
        selected.append(record)
        trends.extend(age_trends(age, expression[:, index], symbol))
    composition = [{"feature": name, "n_donors": len(age), "same_cohort_age_spearman_descriptive": rank_correlation(age, values)}
                   for name, values in (("CD8_naive_fraction", naive), ("CD8_TEM_fraction", tem))]
    output.mkdir(parents=True, exist_ok=True)
    write_table(output / "selected_genes.csv", selected)
    write_table(output / "gene_age_trends.csv", trends)
    write_table(output / "composition_summary.csv", composition)
    if plots:
        make_figures(output, selected, trends, age, naive, tem)
    bibtex = []
    for reference in reference_doc["references"]:
        bibtex.append(
            f"@article{{{reference['id']},\n"
            f"  author = {{{bibtex_authors(reference['authors'])}}},\n"
            f"  title = {{{reference['title']}}},\n"
            f"  journal = {{{reference['journal']}}},\n"
            f"  year = {{{reference['year']}}},\n"
            f"  volume = {{{reference['volume']}}},\n"
            f"  pages = {{{reference['pages'].replace('-', '--')}}},\n"
            f"  doi = {{{reference['doi']}}},\n"
            f"  url = {{{reference['url']}}}\n}}"
        )
    (output / "references.bib").write_text("\n\n".join(bibtex) + "\n", encoding="utf-8")
    summary = {
        "status": "completed", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_donors": len(age), "selected_gene_symbols": [r["gene_symbol"] for r in selected],
        "selection": "Post-hoc illustrative choice among high-ranking annotated 5/5-fold stable features with traceable primary literature; not exhaustive or a new discovery screen.",
        "statistics": "Same-cohort descriptive Spearman and age-band medians/IQR; no p-values, independent validation, retraining, residual correction or clinical testing.",
        "model_coefficients": "Means across outer-fold training models; conditional standardized coefficients, not marginal age correlations or causal years-of-life effects.",
        "source_sha256": {
            "expression_bundle": sha256(data / "cd8_pseudobulk.npz"),
            "donor_metadata": sha256(data / "donor_metadata.csv"),
            "feature_stability": sha256(training / "elastic_net_feature_stability.csv"),
            "training_summary": sha256(training / "summary.json"),
            "literature_registry": sha256(references_path),
        },
        "n_primary_references": len(reference_doc["references"]),
        "limitations": [
            "LRRN3 literature establishes age associations in blood, not a proven anti-aging mechanism in CD8.",
            "Gene-level CDKN2A is not p16-specific; the original reference reported no age correlation for an aggregate p16/ARF assay.",
            "CD70/CD27 experiments are context-dependent and mainly mouse studies, not proof of natural human CD8 aging.",
            "IGFBP3 serum protein associations with naive CD8 maintenance do not validate a harmful role of CD8 IGFBP3 RNA.",
            "Broad-CD8 pseudobulk mixes composition and cell-intrinsic expression; subset fractions are descriptive caveats, not proven mediation.",
            "Literature concordance and overlapping-fold stability are not causal or clinical validation.",
        ],
        "artifacts": {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in sorted(output.iterdir())},
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"genes": selected, "composition": composition}, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/processed/CD8")
    parser.add_argument("--training", type=Path, default=ROOT / "results/cd8_training")
    parser.add_argument("--references", type=Path, default=ROOT / "docs/CD8_BIOLOGY_REFERENCES.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/cd8_biology")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    generate(args.data, args.training, args.references, args.output, not args.no_plots)
