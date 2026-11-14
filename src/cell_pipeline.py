#!/usr/bin/env python3
"""One shared protocol for CD4/B/NK/MONO assignments; CD8 remains a reference.

cohort is leader-only metadata work. Other actions run ONE specified lineage.
No command implicitly trains all cell types or overwrites previous experiments.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import zipfile
from datetime import datetime, timezone

import h5py
import numpy as np

from preprocess_cd8 import NORMALIZATION_NAME, ROOT, describe, preprocess, read_column, sha256
from train_cd8 import GRIDS, load_data, make_splits, run_benchmark, validate_fixed_splits
from validate_cd8 import validate as validate_preprocessing
from validate_cd8_training import validate as validate_training

DEFAULT_CONFIG = ROOT / "config/cell_types.json"
ENGINE_FILES = ("preprocess_cd8.py", "train_cd8.py", "validate_cd8.py", "validate_cd8_training.py", "cell_pipeline.py")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def read_table(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_table(path, records):
    with Path(path).open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def engine_hashes():
    # Normalize text newlines so approved Windows checkouts can share a protocol.
    return {name: hashlib.sha256((ROOT / "src" / name).read_text(encoding="utf-8").encode()).hexdigest()
            for name in ENGINE_FILES}


def validate_config(config):
    if set(config["cells"]) != {"CD4", "B", "NK", "MONO", "CD8"}:
        raise ValueError("The common protocol requires exactly five declared lineages")
    all_labels = [label for cell in config["cells"].values() for label in cell["subtypes"]]
    if len(set(all_labels)) != len(all_labels):
        raise ValueError("A subtype is assigned to more than one lineage")
    if config["training"]["outer_folds"] != 5 or config["training"]["inner_folds"] != 3:
        raise ValueError("The team protocol fixes 5 outer and 3 inner folds")


def serialize_splits(splits):
    return [[train.tolist(), test.tolist()] for train, test in splits]


def build_cohort(input_path, config, shared_dir, public_summary, check_source=True, min_common=50):
    """Freeze eligibility and all split indices from metadata, not expression."""
    validate_config(config)
    input_path, shared_dir, public_summary = map(Path, (input_path, shared_dir, public_summary))
    if shared_dir.exists() and any(shared_dir.iterdir()):
        raise FileExistsError("Shared cohort already exists; do not regenerate independently")
    if public_summary.exists():
        raise FileExistsError("Public cohort record already exists")
    input_hash = sha256(input_path)
    if check_source and (input_path.stat().st_size != config["source_bytes"] or input_hash != config["source_sha256"]):
        raise ValueError("Raw dataset size/hash differs from the frozen source")
    with h5py.File(input_path, "r") as f:
        obs = f["obs"]
        donor = read_column(obs, "donor_id").astype(str)
        names, first, codes = np.unique(donor, return_index=True, return_inverse=True)
        age = read_column(obs, "age").astype(float)
        sex = read_column(obs, "sex").astype(str)
        pool = read_column(obs, "pool_number").astype(str)
        if not np.isfinite(age).all():
            raise ValueError("Missing age")
        for key, values in (("age", age), ("sex", sex), ("pool", pool)):
            if not np.array_equal(values, values[first][codes]):
                raise ValueError(f"Inconsistent donor {key}")
        labels = read_column(obs, config["annotation_column"]).astype(str)
        n_genes = read_column(obs, "nFeature_RNA").astype(float)
        mt = read_column(obs, "percent.mt").astype(float)
        umi = read_column(obs, "nCount_RNA").astype(float)
        qc = config["qc"]
        good = (np.isfinite(n_genes) & (n_genes >= qc["min_genes"])
                & np.isfinite(mt) & (mt >= 0) & (mt <= qc["max_mt"])
                & np.isfinite(umi) & (umi > 0))
        available_labels = set(labels)
        counts = {}
        cell_stats = {}
        for key, spec in config["cells"].items():
            if not set(spec["subtypes"]).issubset(available_labels):
                raise ValueError(f"Configured labels absent from source: {key}")
            selected = np.isin(labels, spec["subtypes"]) & good
            counts[key] = np.bincount(codes[selected], minlength=len(names))
            eligible = counts[key] >= qc["min_cells"]
            cell_stats[key] = {"qc_cells": int(selected.sum()), "eligible_donors": int(eligible.sum()),
                               "cells_in_individually_eligible_donors": int(counts[key][eligible].sum())}
    common = np.logical_and.reduce([c >= qc["min_cells"] for c in counts.values()])
    if int(common.sum()) < min_common:
        raise ValueError("Too few common donors; contact leader instead of weakening QC")
    shared_names, shared_age, shared_pool = names[common], age[first][common], pool[first][common]
    dummy_X = np.zeros((len(shared_names), 1))
    train = config["training"]
    plans = {}
    fold_vectors = {}
    for protocol, grouped in (("donor_cv", False), ("pool_cv", True)):
        outer = make_splits(dummy_X, shared_age, shared_pool, train["outer_folds"], train["seed"], grouped)
        inner = []
        fold_of = np.full(len(shared_names), -1, dtype=int)
        for fold, (tr, te) in enumerate(outer):
            fold_of[te] = fold
            inner.append(serialize_splits(make_splits(dummy_X[tr], shared_age[tr], shared_pool[tr],
                                                     train["inner_folds"], train["seed"] + fold + 1, grouped)))
        fold_vectors[protocol] = fold_of
        plans[protocol] = {"outer": serialize_splits(outer), "inner": inner}
    plans["demo_inner"] = serialize_splits(make_splits(dummy_X, shared_age, shared_pool,
                                                       train["inner_folds"], train["seed"] + 100, False))
    eligibility = []
    common_records = []
    position = 0
    for d, name in enumerate(names):
        record = {"donor_id": name, "age": float(age[first[d]]), "sex": sex[first[d]], "pool_number": pool[first[d]],
                  **{f"n_{cell.lower()}_cells": int(counts[cell][d]) for cell in config["cells"]},
                  "common_eligible": int(common[d])}
        eligibility.append(record)
        if common[d]:
            common_records.append({**record, "donor_cv_fold": int(fold_vectors["donor_cv"][position]),
                                   "pool_cv_fold": int(fold_vectors["pool_cv"][position])})
            position += 1
    for key in cell_stats:
        cell_stats[key]["cells_in_common_cohort"] = int(counts[key][common].sum())
    shared_dir.mkdir(parents=True, exist_ok=True)
    public_summary.parent.mkdir(parents=True, exist_ok=True)
    write_table(shared_dir / "eligibility.csv", eligibility)
    write_table(shared_dir / "common_cohort.csv", common_records)
    write_json(shared_dir / "splits.json", plans)
    protocol = {
        "protocol_id": config["protocol_id"], "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": input_hash, "config_sha256": canonical_digest(config), "engine_sha256": engine_hashes(),
        "cohort_sha256": sha256(shared_dir / "common_cohort.csv"), "splits_sha256": sha256(shared_dir / "splits.json"),
        "n_common_donors": len(shared_names), "training": train, "grids": GRIDS,
        "state": "Metadata eligibility and splits frozen; each lineage must still pass actual expression-count validation before training.",
    }
    write_json(shared_dir / "protocol.json", protocol)
    summary = {
        "protocol_id": config["protocol_id"], "observed_donors": len(names), "n_common_donors": len(shared_names),
        "n_common_pools": len(np.unique(shared_pool)), "common_age": describe(shared_age),
        "common_sex_counts": {s: int(np.sum(sex[first][common] == s)) for s in np.unique(sex)},
        "cell_eligibility_metadata_only": cell_stats, "source_sha256": input_hash,
        "config_sha256": protocol["config_sha256"], "cohort_sha256": protocol["cohort_sha256"],
        "splits_sha256": protocol["splits_sha256"],
        "note": "No expression matrices processed and no models trained by cohort action. Old 977-donor CD8 scores are not fair common-cohort comparators.",
    }
    write_json(public_summary, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def shared_protocol(config, shared_dir):
    shared_dir = Path(shared_dir)
    frozen = read_json(shared_dir / "protocol.json")
    if frozen["config_sha256"] != canonical_digest(config) or frozen["grids"] != GRIDS:
        raise ValueError("Configuration differs from shared protocol; contact leader")
    if frozen["engine_sha256"] != engine_hashes():
        raise ValueError("Shared code differs from the leader's frozen version; do not edit core methods independently")
    for filename, key in (("common_cohort.csv", "cohort_sha256"), ("splits.json", "splits_sha256")):
        if sha256(shared_dir / filename) != frozen[key]:
            raise ValueError("Shared donor/fold manifest checksum mismatch")
    if frozen["source_sha256"] != config["source_sha256"]:
        raise ValueError("Shared source dataset differs from assignment")
    cohort = read_table(shared_dir / "common_cohort.csv")
    if len(cohort) != frozen["n_common_donors"] or len({r["donor_id"] for r in cohort}) != len(cohort):
        raise ValueError("Invalid shared cohort")
    return frozen, cohort, read_json(shared_dir / "splits.json")


def cell_paths(cell):
    return {
        "data": ROOT / "data/processed" / cell,
        "preprocessing": ROOT / "results/cd8" if cell == "CD8" else ROOT / "results/cells" / cell / "preprocessing",
        "models": ROOT / "data/models" / cell / "common_v1",
        "report": ROOT / "results/cells" / cell / "common_v1",
        "bundle": f"{cell.lower()}_pseudobulk.npz",
    }


def check_prepared_protocol(cell, config, source):
    params = source["parameters"]
    qc = config["qc"]
    if (source["input"]["sha256"] != config["source_sha256"]
            or params["subtypes"] != config["cells"][cell]["subtypes"]
            or params["min_cells_per_donor"] != qc["min_cells"]
            or params["min_source_genes_per_cell"] != qc["min_genes"]
            or params["max_source_mt_percent"] != qc["max_mt"]
            or params.get("cell_key", "CD8") != cell
            or params.get("bundle_name", "cd8_pseudobulk.npz") != f"{cell.lower()}_pseudobulk.npz"
            or params["subtype_column"] != config["annotation_column"]
            or params.get("annotation_score_filter") is not None
            or params["normalization"] != NORMALIZATION_NAME):
        raise ValueError("Prepared data source, labels or QC differ from the shared protocol")


def prepare_cell(cell, config, input_path, paths):
    summary_path = paths["preprocessing"] / "summary.json"
    if summary_path.exists():
        old = read_json(summary_path)
        check_prepared_protocol(cell, config, old)
        for name, info in old["artifacts"].items():
            if sha256(paths["data"] / name) != info["sha256"]:
                raise ValueError("Existing prepared-data checksum mismatch")
        validate_preprocessing(Path(input_path), paths["data"], paths["preprocessing"])
        print(f"Reusing verified existing {cell} preprocessing; no overwrite.")
        return
    qc = config["qc"]
    if Path(input_path).stat().st_size != config["source_bytes"]:
        raise ValueError("Raw input size differs from official dataset")
    summary = preprocess(Path(input_path), paths["data"], paths["preprocessing"],
                         min_cells=qc["min_cells"], min_genes=qc["min_genes"], max_mt=qc["max_mt"],
                         subtypes=config["cells"][cell]["subtypes"], cell_key=cell)
    check_prepared_protocol(cell, config, summary)
    validate_preprocessing(Path(input_path), paths["data"], paths["preprocessing"])


def train_cell(cell, config, shared_dir, paths):
    frozen, cohort, plans = shared_protocol(config, shared_dir)
    X, y, ids, genes, symbols, groups, source = load_data(paths["data"], paths["preprocessing"] / "summary.json", paths["bundle"])
    check_prepared_protocol(cell, config, source)
    prepared_validation = read_json(paths["preprocessing"] / "validation.json")
    if prepared_validation.get("status") != "passed" or not prepared_validation.get("independent_source_spot_checks"):
        raise ValueError("Obtain source-verified preprocessing from the leader or run prepare first")
    lookup = {name: i for i, name in enumerate(ids)}
    wanted = [r["donor_id"] for r in cohort]
    if not set(wanted).issubset(lookup):
        raise ValueError("This lineage is missing shared donors; contact leader, do not silently delete people")
    indices = np.array([lookup[name] for name in wanted], dtype=int)
    X, y, ids, groups = X[indices], y[indices], ids[indices], groups[indices]
    if not np.array_equal(y, [float(r["age"]) for r in cohort]) or not np.array_equal(groups, [r["pool_number"] for r in cohort]):
        raise ValueError("Shared and prepared age/pool metadata differ")
    meta = {r["donor_id"]: r for r in read_table(paths["data"] / "donor_metadata.csv")}
    for r in cohort:
        count_key = f"n_{cell.lower()}_cells"
        if int(meta[r["donor_id"]][count_key]) != int(r[count_key]):
            raise ValueError("Prepared cell counts disagree with shared eligibility")
    records = [{**meta[name], "donor_cv_fold": cohort[i]["donor_cv_fold"], "pool_cv_fold": cohort[i]["pool_cv_fold"]}
               for i, name in enumerate(wanted)]
    outer = {p: plans[p]["outer"] for p in ("donor_cv", "pool_cv")}
    inner = {p: plans[p]["inner"] for p in ("donor_cv", "pool_cv")}
    t = config["training"]
    return run_benchmark(X, y, ids, genes, symbols, groups, paths["models"], paths["report"],
                         seed=t["seed"], max_features=t["max_features"], outer_folds=t["outer_folds"],
                         inner_folds=t["inner_folds"], max_iter=t["max_iter"], source=source,
                         cell_label=cell, fixed_outer_splits=outer, fixed_inner_splits=inner,
                         demo_inner_splits=plans["demo_inner"], cohort_records=records,
                         run_metadata={"cell_key": cell, "protocol_id": config["protocol_id"],
                                       "common_cohort_sha256": frozen["cohort_sha256"],
                                       "splits_sha256": frozen["splits_sha256"], "config_sha256": frozen["config_sha256"],
                                       "prepared_bundle_sha256": source["artifacts"][paths["bundle"]]["sha256"],
                                       "eligible_donors_before_common": source["donors"]["retained"]})


def validate_cell(cell, config, shared_dir, paths, input_path):
    frozen, cohort, plans = shared_protocol(config, shared_dir)
    check_prepared_protocol(cell, config, read_json(paths["preprocessing"] / "summary.json"))
    if Path(input_path).is_file():
        validate_preprocessing(Path(input_path), paths["data"], paths["preprocessing"])
    else:
        original = read_json(paths["preprocessing"] / "validation.json")
        if original.get("status") != "passed" or not original.get("independent_source_spot_checks"):
            raise ValueError("Raw input absent and no leader source-verification record supplied")
        validate_preprocessing(Path(input_path), paths["data"], paths["preprocessing"],
                               source_spot_checks=False, destination_name="artifact_validation.json")
    validate_training(paths["data"], paths["models"], paths["report"], paths["bundle"])
    summary = read_json(paths["report"] / "summary.json")
    metadata = summary.get("run_metadata", {})
    prepared = read_json(paths["preprocessing"] / "summary.json")
    if (summary["n_donors"] != len(cohort) or metadata.get("common_cohort_sha256") != frozen["cohort_sha256"]
            or summary.get("cell_label") != cell or metadata.get("cell_key") != cell
            or metadata.get("splits_sha256") != frozen["splits_sha256"]
            or metadata.get("config_sha256") != frozen["config_sha256"]
            or metadata.get("prepared_bundle_sha256") != prepared["artifacts"][paths["bundle"]]["sha256"]):
        raise ValueError("Training lineage, source data or common protocol does not match")
    actual_cohort = read_table(paths["models"] / "analysis_cohort.csv")
    if [r["donor_id"] for r in actual_cohort] != [r["donor_id"] for r in cohort]:
        raise ValueError("Analysis-cohort row order differs from frozen donor order")
    for observed, expected in zip(actual_cohort, cohort):
        if (float(observed["age"]) != float(expected["age"])
                or observed["pool_number"] != expected["pool_number"]
                or int(observed[f"n_{cell.lower()}_cells"]) != int(expected[f"n_{cell.lower()}_cells"])):
            raise ValueError("Analysis-cohort metadata differs from frozen metadata")
    settings, approved = summary["configuration"], config["training"]
    for actual_key, approved_key in (("seed", "seed"), ("outer_folds", "outer_folds"),
                                     ("inner_folds", "inner_folds"), ("max_features", "max_features"),
                                     ("max_iter", "max_iter")):
        if settings[actual_key] != approved[approved_key]:
            raise ValueError("Saved training settings differ from frozen configuration")
    if settings["grids"] != GRIDS:
        raise ValueError("Saved tuning grid differs from frozen protocol")
    actual = read_table(paths["models"] / "fold_assignments.csv")
    for protocol in ("donor_cv", "pool_cv"):
        expected = {r["donor_id"]: int(r[f"{protocol}_fold"]) for r in cohort}
        observed = {r["donor_id"]: int(r["fold"]) for r in actual if r["protocol"] == protocol}
        if expected != observed:
            raise ValueError("Saved OOF assignments differ from approved common folds")
    result = {"status": "passed", "cell_key": cell, "common_donors": len(cohort),
              "cohort_sha256": frozen["cohort_sha256"], "splits_sha256": frozen["splits_sha256"],
              "config_sha256": frozen["config_sha256"], "shared_outer_inner_protocol_verified": True,
              "cohort_metadata_against_shared_verified": True, "training_settings_verified": True}
    target = paths["report"] / "common_protocol_validation.json"
    if target.exists():
        if read_json(target) != result: raise ValueError("Protocol validation changed")
    else: write_json(target, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def package_cell(cell, config, shared_dir, paths, input_path, deliverables):
    validate_cell(cell, config, shared_dir, paths, input_path)
    for name, minimum in (("handoff_summary.md", 150), ("biological_interpretation.md", 250)):
        path = paths["report"] / name
        if not path.exists(): raise FileNotFoundError(f"Member must write {path} before final packaging")
        text = path.read_text(encoding="utf-8")
        if len(text.strip()) < minimum or any(marker in text for marker in ("TODO", "待填写", "<填写", "TBD")):
            raise ValueError(f"Incomplete member narrative: {path}")
    deliverables = Path(deliverables)
    deliverables.mkdir(parents=True, exist_ok=True)
    destination = deliverables / f"{cell}_return.zip"
    if destination.exists(): raise FileExistsError("Return archive exists; preserve previous version")
    files = []
    for directory, prefix in ((paths["preprocessing"], "preprocessing"), (paths["report"], "analysis")):
        for path in directory.iterdir():
            if path.is_file() and path.suffix.lower() in {".md", ".json", ".csv", ".png", ".svg", ".bib"}:
                files.append((path, f"{cell}/{prefix}/{path.name}"))
    for name in ("donor_metadata.csv", "donor_qc.csv", "gene_metadata.csv"):
        files.append((paths["data"] / name, f"{cell}/private/prepared/{name}"))
    for name in ("oof_predictions.csv", "fold_assignments.csv", "analysis_cohort.csv", "inner_tuning_scores.csv",
                 "fit_diagnostics.json", "elastic_net_demo_model.npz"):
        files.append((paths["models"] / name, f"{cell}/private/model/{name}"))
    files.append((Path(shared_dir) / "protocol.json", f"{cell}/protocol.json"))
    for name in ("common_cohort.csv", "splits.json"):
        files.append((Path(shared_dir) / name, f"{cell}/private/shared/{name}"))
    config_payload = (json.dumps(config, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    for source, _ in files:
        if not source.is_file(): raise FileNotFoundError(source)
    manifest = {"cell_key": cell, "protocol_id": config["protocol_id"],
                "notice": "Contains participant-level derived data. Send privately to the group leader; do not publish this ZIP.",
                "files": {name: {"sha256": sha256(path), "bytes": path.stat().st_size} for path, name in files}}
    manifest["files"][f"{cell}/cell_types.json"] = {"sha256": hashlib.sha256(config_payload).hexdigest(), "bytes": len(config_payload)}
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path, name in files: archive.write(path, name)
        archive.writestr(f"{cell}/cell_types.json", config_payload)
        archive.writestr(f"{cell}/RETURN_MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"Return archive: {destination} ({destination.stat().st_size:,} bytes); private sharing only.")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("cohort", "prepare", "train", "validate", "package"))
    parser.add_argument("--cell", choices=("CD4", "B", "NK", "MONO", "CD8"))
    parser.add_argument("--input", type=Path, default=ROOT / "dataset/onek1k.h5ad")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--shared", type=Path, default=ROOT / "data/shared")
    args = parser.parse_args()
    config = read_json(args.config)
    validate_config(config)
    if args.action == "cohort":
        build_cohort(args.input, config, args.shared, ROOT / "results/shared/common_cohort_summary.json")
        return
    if not args.cell: parser.error("--cell is required except for the leader-only cohort action")
    shared_protocol(config, args.shared)
    paths = cell_paths(args.cell)
    if args.action == "prepare": prepare_cell(args.cell, config, args.input, paths)
    elif args.action == "train": train_cell(args.cell, config, args.shared, paths)
    elif args.action == "validate": validate_cell(args.cell, config, args.shared, paths, args.input)
    else: package_cell(args.cell, config, args.shared, paths, args.input, ROOT / "deliverables")


if __name__ == "__main__":
    main()
