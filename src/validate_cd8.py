#!/usr/bin/env python3
"""Independently validate saved CD8 files and spot-check source donor counts."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from preprocess_cd8 import ROOT, read_column


def load_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def validate(input_path, output_dir, report_dir, source_spot_checks=True, destination_name="validation.json"):
    summary = json.loads((report_dir / "summary.json").read_text(encoding="utf-8"))
    cell_key = summary["parameters"].get("cell_key", "CD8")
    slug = cell_key.lower()
    bundle_name = summary["parameters"].get("bundle_name", f"{slug}_pseudobulk.npz")
    for name, artifact in summary["artifacts"].items():
        digest = hashlib.sha256()
        with (output_dir / name).open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != artifact["sha256"]:
            raise ValueError(f"Artifact checksum mismatch: {name}")
    with np.load(output_dir / bundle_name, allow_pickle=False) as bundle:
        counts, expression = bundle["counts"], bundle["log1p_cpm"]
        donors, genes = bundle["donor_ids"], bundle["gene_ids"]
    metadata = load_csv(output_dir / "donor_metadata.csv")
    gene_metadata = load_csv(output_dir / "gene_metadata.csv")
    if donors.tolist() != [r["donor_id"] for r in metadata]:
        raise ValueError("Donor row alignment failed")
    if genes.tolist() != [r["gene_id"] for r in gene_metadata]:
        raise ValueError("Gene column alignment failed")
    if counts.shape != expression.shape or list(counts.shape) != summary["validation"]["output_shape"]:
        raise ValueError("Output shape mismatch")
    if counts.dtype != np.int64 or np.any(counts < 0):
        raise ValueError("Invalid saved counts")
    if not np.isfinite(expression).all() or np.any(expression < 0):
        raise ValueError("Invalid saved normalized expression")
    library = counts.sum(axis=1, dtype=np.int64)
    np.testing.assert_array_equal(library, [int(r[f"actual_{slug}_umi"]) for r in metadata])
    if int(library.sum()) != summary["validation"]["raw_integer_count_sum"]:
        raise ValueError("Saved total count mismatch")
    for index in range(len(donors)):
        expected = np.log1p(counts[index].astype(float) / library[index] * 1e6)
        np.testing.assert_allclose(expression[index], expected, rtol=1e-6, atol=1e-6)
    for row in metadata:
        if int(row[f"n_{slug}_cells"]) < summary["parameters"]["min_cells_per_donor"]:
            raise ValueError("Included donor violates minimum cell count")
        fractions = [float(v) for key, v in row.items() if key.startswith("fraction_")]
        np.testing.assert_allclose(sum(fractions), 1, rtol=1e-12)
    # A different aggregation path: directly read each cell of three whole donors.
    # This is a deterministic spot-check, not another full-cohort preprocessing run.
    checked = []
    if source_spot_checks:
        with h5py.File(input_path, "r") as f:
            donor_per_cell = read_column(f["obs"], "donor_id").astype(str)
            labels = read_column(f["obs"], "predicted.celltype.l2").astype(str)
            features = read_column(f["obs"], "nFeature_RNA").astype(float)
            mt = read_column(f["obs"], "percent.mt").astype(float)
            umi = read_column(f["obs"], "nCount_RNA").astype(float)
            p = summary["parameters"]
            mask = (np.isin(labels, p["subtypes"]) & np.isfinite(features)
                    & (features >= p["min_source_genes_per_cell"])
                    & np.isfinite(mt) & (mt >= 0) & (mt <= p["max_source_mt_percent"])
                    & np.isfinite(umi) & (umi > 0))
            ptr = f["X/indptr"][:]
            for d in sorted(set([0, len(donors) // 2, len(donors) - 1])):
                rows = np.flatnonzero(mask & (donor_per_cell == donors[d]))
                independent = np.zeros(len(genes), dtype=np.int64)
                for cell in rows:
                    lo, hi = int(ptr[cell]), int(ptr[cell + 1])
                    np.add.at(independent, f["X/indices"][lo:hi], f["X/data"][lo:hi].astype(np.int64))
                np.testing.assert_array_equal(independent, counts[d])
                if len(rows) != int(metadata[d][f"n_{slug}_cells"]):
                    raise ValueError("Independent donor cell count mismatch")
                checked.append({"output_row_index": int(d), "cells": int(len(rows)), "exact_count_match": True})
    from PIL import Image
    with Image.open(report_dir / "qc_overview.png") as image:
        image.verify()
    result = {
        "status": "passed", "artifact_sha256_verified": True,
        "donor_gene_order_verified": True, "all_saved_normalization_rows_verified": len(donors),
        "independent_source_spot_checks": checked, "qc_png_decodes": True,
        "scope": "File/data processing validation only; no age-model performance evaluated.",
    }
    if not source_spot_checks:
        result["source_access_mode"] = "artifact-only; original-source checks must be supplied by the leader"
    destination = report_dir / destination_name
    if destination.exists():
        previous = json.loads(destination.read_text(encoding="utf-8"))
        if previous != result:
            raise ValueError("Validation result changed; use a new report directory rather than overwriting provenance")
    else:
        with destination.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "dataset/onek1k.h5ad")
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/CD8")
    parser.add_argument("--report", type=Path, default=ROOT / "results/cd8")
    args = parser.parse_args()
    validate(args.input, args.output, args.report)
