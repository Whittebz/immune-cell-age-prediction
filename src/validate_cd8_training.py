#!/usr/bin/env python3
"""Recompute saved OOF metrics and verify donor/pool splits and baselines."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from preprocess_cd8 import ROOT, sha256
from train_cd8 import metrics, predict_numeric_model


def csv_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def validate(data, output, report, bundle_name="cd8_pseudobulk.npz"):
    summary = json.loads((report / "summary.json").read_text(encoding="utf-8"))
    for name, artifact in summary["private_artifacts"].items():
        if sha256(output / name) != artifact["sha256"]:
            raise ValueError(f"Artifact checksum mismatch: {name}")
    source_meta = csv_rows(data / "donor_metadata.csv")
    cohort_path = output / "analysis_cohort.csv"
    meta = csv_rows(cohort_path) if cohort_path.exists() else source_meta
    source_lookup = {r["donor_id"]: r for r in source_meta}
    if len({r["donor_id"] for r in meta}) != len(meta):
        raise ValueError("Duplicate analysis-cohort IDs")
    for row in meta:
        original = source_lookup[row["donor_id"]]
        if float(row["age"]) != float(original["age"]) or row["pool_number"] != original["pool_number"]:
            raise ValueError("Analysis-cohort metadata differs from source")
    if len(meta) != summary["n_donors"]:
        raise ValueError("Analysis-cohort size differs from training summary")
    all_ids = {r["donor_id"] for r in meta}
    age = {r["donor_id"]: float(r["age"]) for r in meta}
    pool = {r["donor_id"]: r["pool_number"] for r in meta}
    assignments = csv_rows(output / "fold_assignments.csv")
    oof = csv_rows(output / "oof_predictions.csv")
    fold_scores = csv_rows(report / "fold_metrics.csv")
    score_rows = csv_rows(report / "metrics.csv")
    recomputed = []
    for protocol in ("donor_cv", "pool_cv"):
        selected = [r for r in assignments if r["protocol"] == protocol]
        if len(selected) != len(meta) or {r["donor_id"] for r in selected} != all_ids:
            raise ValueError("Incomplete/duplicate outer-fold mapping")
        fold_of = {r["donor_id"]: int(r["fold"]) for r in selected}
        for fold in range(summary["configuration"]["outer_folds"]):
            test_ids = {i for i in all_ids if fold_of[i] == fold}
            train_ids = all_ids - test_ids
            if protocol == "pool_cv" and {pool[i] for i in train_ids} & {pool[i] for i in test_ids}:
                raise ValueError("Pool leakage in held-out-pool evaluation")
            baseline = float(np.median([age[i] for i in train_ids]))
            baseline_rows = [r for r in oof if r["protocol"] == protocol and r["model"] == "median" and int(r["fold"]) == fold]
            if {r["donor_id"] for r in baseline_rows} != test_ids:
                raise ValueError("Baseline test-fold coverage mismatch")
            np.testing.assert_allclose([float(r["predicted_age"]) for r in baseline_rows], baseline, atol=1e-12)
        names = ["median", "ridge", "elastic_net"] if protocol == "donor_cv" else ["median", "ridge"]
        baseline_errors = {r["donor_id"]: abs(float(r["predicted_age"]) - age[r["donor_id"]])
                           for r in oof if r["protocol"] == protocol and r["model"] == "median"}
        for name in names:
            rows = [r for r in oof if r["protocol"] == protocol and r["model"] == name]
            if len(rows) != len(meta) or {r["donor_id"] for r in rows} != all_ids:
                raise ValueError("Incomplete/duplicate OOF predictions")
            for row in rows:
                donor = row["donor_id"]
                if float(row["actual_age"]) != age[donor] or int(row["fold"]) != fold_of[donor]:
                    raise ValueError("OOF age/fold differs from source metadata")
                np.testing.assert_allclose(float(row["raw_age_gap"]), float(row["predicted_age"]) - age[donor])
            target = np.array([age[r["donor_id"]] for r in rows])
            prediction = np.array([float(r["predicted_age"]) for r in rows])
            measured = metrics(target, prediction, name != "median")
            saved = next(r for r in score_rows if r["protocol"] == protocol and r["model"] == name)
            for key, value in measured.items():
                if value is None:
                    if saved[key] != "":
                        raise ValueError("Inapplicable correlation should be empty")
                else:
                    np.testing.assert_allclose(float(saved[key]), value, atol=1e-10)
            delta = np.mean([abs(float(r["predicted_age"]) - age[r["donor_id"]]) - baseline_errors[r["donor_id"]] for r in rows])
            np.testing.assert_allclose(float(saved["mae_delta_vs_median_years"]), delta, atol=1e-10)
            for fold in range(summary["configuration"]["outer_folds"]):
                part = [r for r in rows if int(r["fold"]) == fold]
                fm = metrics(np.array([float(r["actual_age"]) for r in part]),
                             np.array([float(r["predicted_age"]) for r in part]), name != "median")
                recorded = next(r for r in fold_scores if r["protocol"] == protocol and r["model"] == name and int(r["fold"]) == fold)
                for key, value in fm.items():
                    if value is not None:
                        np.testing.assert_allclose(float(recorded[key]), value, atol=1e-10)
            recomputed.append({"protocol": protocol, "model": name, "mae_years": measured["mae_years"]})
    diagnostics = json.loads((output / "fit_diagnostics.json").read_text(encoding="utf-8"))
    if len(diagnostics) != summary["n_recorded_fits"]:
        raise ValueError("Missing fit diagnostics")
    if sum(r["convergence_warning_count"] for r in diagnostics) != summary["convergence_warning_count_all_fits"]:
        raise ValueError("Convergence warning summary mismatch")
    with np.load(data / bundle_name, allow_pickle=False) as bundle:
        demo = predict_numeric_model(bundle["log1p_cpm"][:3], bundle["gene_ids"], output / "elastic_net_demo_model.npz")
    if not np.isfinite(demo).all():
        raise ValueError("Demo artifact produced nonfinite predictions")
    result = {
        "status": "passed", "unique_donors": len(meta), "oof_rows_checked": len(oof),
        "donor_and_pool_fold_coverage_verified": True, "held_out_pool_disjointness_verified": True,
        "train_fold_median_baseline_verified": True, "aggregate_and_fold_metrics_recomputed": True,
        "artifact_hashes_verified": True, "convergence_logs_consistent": True,
        "numeric_demo_artifact_loads_without_pickle": True, "recomputed_metrics": recomputed,
        "scope": "Consistency checks of saved results; not independent cohort validation or proof of biology.",
    }
    destination = report / "validation.json"
    if destination.exists():
        if json.loads(destination.read_text(encoding="utf-8")) != result:
            raise ValueError("Validation changed; preserve prior record and investigate")
    else:
        with destination.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/processed/CD8")
    parser.add_argument("--output", type=Path, default=ROOT / "data/models/CD8")
    parser.add_argument("--report", type=Path, default=ROOT / "results/cd8_training")
    args = parser.parse_args()
    validate(args.data, args.output, args.report)
