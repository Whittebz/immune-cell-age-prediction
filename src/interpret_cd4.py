#!/usr/bin/env python3
"""Describe CD4 common-cohort model features and composition-adjusted associations.

Reads frozen training outputs; writes aggregate tables/figures without retraining.
The three representatives are post-hoc examples chosen for stability and traceable
literature, not the three largest coefficients or an independent validation set.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

from audit_cd8_features import partial_rank
from preprocess_cd8 import ROOT, sha256
from train_cd8 import load_data, export_figure

CANDIDATES = (
    ("LRRN3", "ENSG00000173114", "R1", "年龄相关血液表达"),
    ("CD70", "ENSG00000125726", "R2", "CD4 T细胞共刺激"),
    ("GZMB", "ENSG00000100453", "R3", "细胞毒性CD4 T细胞"),
)
PARTS = ("cd4_naive", "cd4_tcm", "cd4_tem", "cd4_ctl", "cd4_proliferating", "treg")
# Fractions sum to one; TEM is the explicit omitted reference component.
COMPOSITION_COLUMNS = (0, 1, 3, 4, 5)
AGE_BINS = ((18, 40), (40, 60), (60, 80), (80, 101))


def read_table(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_table(path, records):
    with Path(path).open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def align_common(ids, metadata, cohort):
    """Align by ID, require exact age/pool/composition metadata agreement."""
    if len(set(ids)) != len(ids) or len(set(r["donor_id"] for r in cohort)) != len(cohort):
        raise ValueError("Duplicate donor identifiers")
    if [r["donor_id"] for r in metadata] != list(ids):
        raise ValueError("Prepared metadata order differs from expression")
    lookup = {name: i for i, name in enumerate(ids)}
    if not set(r["donor_id"] for r in cohort).issubset(lookup):
        raise ValueError("Common cohort contains a donor absent from prepared expression")
    indices = np.array([lookup[r["donor_id"]] for r in cohort], dtype=int)
    fields = ("age", "sex", "pool_number", "n_cd4_cells", "actual_cd4_umi")
    fields += tuple("fraction_" + p for p in PARTS)
    for index, target in zip(indices, cohort):
        source = metadata[index]
        for key in fields:
            if source[key] != target[key]:
                raise ValueError("Common and prepared metadata differ: " + key)
    return indices


def gene_index(genes, symbols, gene_id, symbol):
    found = np.flatnonzero(np.asarray(genes) == gene_id)
    if len(found) != 1 or str(symbols[found[0]]) != symbol:
        raise ValueError("Gene ID/symbol is absent or ambiguous: " + symbol)
    return int(found[0])


def composition_matrix(cohort):
    values = np.array([[float(r["fraction_" + p]) for p in PARTS] for r in cohort])
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
        raise ValueError("Invalid CD4 cell fractions")
    np.testing.assert_allclose(values.sum(axis=1), 1.0, atol=1e-10)
    return values


def association(age, expression, composition):
    if np.std(age) == 0 or np.std(expression) == 0:
        raise ValueError("Age and expression must vary")
    marginal = float(spearmanr(age, expression).statistic)
    covariates = np.column_stack([rankdata(composition[:, j]) for j in COMPOSITION_COLUMNS])
    adjusted, rank = partial_rank(expression, age, covariates)
    if adjusted is None:
        raise ValueError("Composition adjustment leaves degenerate residuals")
    return marginal, adjusted, rank


def figures(output, selected, trends):
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 13, "axes.spines.top": False, "axes.spines.right": False})
    names = [r["gene_symbol"] for r in selected]
    coefs = [r["donor_cv_mean_coefficient"] for r in selected]
    fig, ax = plt.subplots(figsize=(8.8, 4.5), constrained_layout=True)
    ax.barh(names, coefs, color=["#287C8E" if c < 0 else "#B7743B" for c in coefs], height=.55)
    ax.axvline(0, color="#65788E", lw=1)
    ax.invert_yaxis()
    ax.set(xlim=(-1.5, 1.0), xlabel="Mean standardized model coefficient (years)", title="CD4 T | common 505 donors | 5/5 nonzero, same sign")
    for i, value in enumerate(coefs):
        ax.text(value + (.04 if value >= 0 else -.04), i, f"{value:+.3f}", va="center", ha="left" if value >= 0 else "right", fontsize=13)
    export_figure(fig, output / "selected_gene_coefficients.png");plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    pos = np.arange(len(names));height = .32
    ax.barh(pos-height/2, [r["age_spearman"] for r in selected], height, color="#1A448F", label="Unadjusted")
    ax.barh(pos+height/2, [r["composition_adjusted_rank_correlation"] for r in selected], height, color="#6BA7B8", label="Adjusted for cell fractions")
    ax.set_yticks(pos, names);ax.invert_yaxis();ax.axvline(0,color="#65788E",lw=1)
    ax.set(xlim=(-.65,.48), xlabel="Age-expression rank correlation", title="CD4 T | same 505 donors | descriptive associations")
    ax.legend(loc="lower left", frameon=False)
    export_figure(fig,output/"composition_adjusted_associations.png");plt.close(fig)
    fig, axes = plt.subplots(1,3,figsize=(12,3.7),constrained_layout=True)
    labels=[f"{lo}–{hi-1}" for lo,hi in AGE_BINS]
    for ax,name in zip(axes,names):
        values=[r for r in trends if r["gene_symbol"]==name]
        med=np.array([r["median_log1p_cpm"] for r in values]);low=np.array([r["q25_log1p_cpm"] for r in values]);high=np.array([r["q75_log1p_cpm"] for r in values]);x=np.arange(len(values))
        ax.plot(x,med,color="#1A448F",marker="o");ax.fill_between(x,low,high,color="#D4E0EF",alpha=.7)
        ax.set(xticks=x,xticklabels=labels,title=name,xlabel="Age (years)")
    axes[0].set_ylabel("log1p CPM (median and IQR)")
    export_figure(fig,output/"gene_age_trends.png");plt.close(fig)


def generate(data, report, models, references, output, make_plots=True):
    data,report,models,references,output=map(Path,(data,report,models,references,output))
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Preserve existing interpretation output: " + str(output))
    training=json.loads((report/"summary.json").read_text())
    if training["cell_label"]!="CD4" or training["n_donors"]!=505:
        raise ValueError("This interpretation requires the frozen CD4 common505 model")
    if json.loads((report/"validation.json").read_text())["status"]!="passed":
        raise ValueError("Validate the frozen CD4 model before interpretation")
    paths=[data/"cd4_pseudobulk.npz",data/"donor_metadata.csv",models/"analysis_cohort.csv",report/"summary.json",report/"validation.json",report/"elastic_net_feature_stability.csv",report.parent/"pool_en_common_v1/elastic_net_feature_stability.csv",references]
    before={str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p):sha256(p) for p in paths}
    X,age,ids,genes,symbols,pools,_=load_data(data,report.parent/"preprocessing/summary.json","cd4_pseudobulk.npz")
    metadata=read_table(data/"donor_metadata.csv");cohort=read_table(models/"analysis_cohort.csv")
    indices=align_common(ids,metadata,cohort)
    if len(indices)!=505:raise ValueError("Unexpected common cohort size")
    X,age=X[indices],age[indices]
    comp=composition_matrix(cohort)
    stability=read_table(report/"elastic_net_feature_stability.csv")
    en={r["gene_id"]:r for r in stability};pool={r["gene_id"]:r for r in read_table(report.parent/"pool_en_common_v1/elastic_net_feature_stability.csv")}
    refs={r["id"]:r for r in json.loads(references.read_text())["references"]}
    selected=[];trends=[]
    for symbol,gid,refid,context in CANDIDATES:
        j=gene_index(genes,symbols,gid,symbol);e=en[gid];p=pool[gid]
        if any(int(r["nonzero_folds"])!=5 or float(r["sign_consistency_when_nonzero"])!=1 for r in [e,p]):
            raise ValueError("Representative is not direction-stable in both protocols")
        if symbol not in refs[refid]["genes"]:raise ValueError("Missing gene-specific reference")
        marginal,adjusted,rank=association(age,X[:,j],comp)
        selected.append({"gene_id":gid,"gene_symbol":symbol,"n_donors":505,"donor_cv_nonzero_folds":int(e["nonzero_folds"]),"pool_cv_nonzero_folds":int(p["nonzero_folds"]),"donor_cv_mean_coefficient":float(e["mean_signed_standardized_coefficient"]),"pool_cv_mean_coefficient":float(p["mean_signed_standardized_coefficient"]),"age_spearman":marginal,"composition_adjusted_rank_correlation":adjusted,"composition_design_rank":rank,"reference_id":refid,"context":context})
        for low,high in AGE_BINS:
            a=X[(age>=low)&(age<high),j]
            if len(a):trends.append({"gene_symbol":symbol,"age_band":f"{low}-{high-1}","n_donors":len(a),"median_log1p_cpm":float(np.median(a)),"q25_log1p_cpm":float(np.percentile(a,25)),"q75_log1p_cpm":float(np.percentile(a,75))})
    output.mkdir(parents=True,exist_ok=True)
    write_table(output/"selected_genes.csv",selected);write_table(output/"gene_age_trends.csv",trends)
    parts=[{"subtype":p,"n_donors":505,"age_spearman":float(spearmanr(age,comp[:,j]).statistic)} for j,p in enumerate(PARTS)]
    write_table(output/"composition_age_correlations.csv",parts)
    if make_plots:figures(output,selected,trends)
    assert all(sha256(p)==before[str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)] for p in paths)
    summary={"status":"completed","cell_type":"CD4 T","cohort":"common_v1","n_donors":505,"n_representatives":3,"selected":selected,"composition":parts,"selection":"Post-hoc stable examples with traceable literature; not top-three ranking or a new model","adjustment":"Residual correlation of ranked age and ranked expression after OLS on intercept plus five ranked cell fractions; CD4 TEM omitted","evidence_scope":"Same-cohort descriptive associations; reference populations and assays recorded separately","source_sha256":before,"artifacts":{p.name:{"sha256":sha256(p),"bytes":p.stat().st_size} for p in output.iterdir() if p.is_file()}}
    (output/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
    (output/"validation.json").write_text(json.dumps({"status":"passed","cohort_rows":505,"metadata_alignment_checked":True,"source_hashes_unchanged":True,"composition_sum_checked":True,"both_protocol_stability_checked":True,"references_checked":3,"scope":"Read-only interpretation checks; frozen model training validation read separately"},indent=2)+"\n")
    print(json.dumps({"n_donors":505,"genes":selected},ensure_ascii=False,indent=2))
    return summary


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,default=ROOT/"data/processed/CD4")
    parser.add_argument("--report",type=Path,default=ROOT/"results/cells/CD4/common_v1")
    parser.add_argument("--models",type=Path,default=ROOT/"data/models/CD4/common_v1")
    parser.add_argument("--references",type=Path,default=ROOT/"docs/CD4_BIOLOGY_REFERENCES.json")
    parser.add_argument("--output",type=Path,default=ROOT/"results/cd4_biology_common505_verified")
    parser.add_argument("--no-plots",action="store_true")
    args=parser.parse_args()
    generate(args.data,args.report,args.models,args.references,args.output,not args.no_plots)
