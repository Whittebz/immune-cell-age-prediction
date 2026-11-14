"""Small synthetic fixtures test mechanics only; never biological results."""
import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from preprocess_cd8 import normalise_counts, plot_qc, preprocess


def fixture(path: Path, fractional: bool = False, conflicting_age: bool = False) -> None:
    matrix = np.array([
        [1, 0, 2, 0], [0, 2, 0, 0], [3, 0, 0, 1],
        [0, 4, 0, 0], [2, 0, 1, 0], [1, 1, 0, 0],
        [0, 0, 5, 0], [5, 0, 0, 0],
    ], dtype=np.float32)
    if fractional:
        matrix[0, 0] = 0.5
    rows, cols = np.nonzero(matrix)
    values = matrix[rows, cols]
    pointers = np.concatenate(([0], np.cumsum(np.bincount(rows, minlength=len(matrix)))))
    string_dtype = h5py.string_dtype("utf-8")
    with h5py.File(path, "w") as f:
        x = f.create_group("X")
        x.attrs["encoding-type"] = "csr_matrix"
        x.attrs["shape"] = matrix.shape
        x.create_dataset("data", data=values)
        x.create_dataset("indices", data=cols)
        x.create_dataset("indptr", data=pointers)
        obs = f.create_group("obs")
        def strings(name, items):
            categories, codes = np.unique(np.array(items, dtype=str), return_inverse=True)
            group = obs.create_group(name)
            group.create_dataset("codes", data=codes)
            group.create_dataset("categories", data=categories.astype(object), dtype=string_dtype)
        strings("donor_id", ["d1", "d1", "d1", "d2", "d2", "d3", "d4", "d1"])
        strings("sex", ["female"] * 8)
        strings("pool_number", ["1", "1", "1", "2", "2", "2", "3", "1"])
        strings("predicted.celltype.l2", ["CD8 Naive", "CD8 TCM", "CD8 TEM", "CD8 TEM",
                                         "CD8 Proliferating", "CD8 Naive", "B naive", "CD8 Naive"])
        ages = [25, 25, 25, 60, 60, 75, 40, 25]
        if conflicting_age:
            ages[1] = 26
        obs.create_dataset("age", data=ages)
        obs.create_dataset("nFeature_RNA", data=(matrix > 0).sum(axis=1))
        obs.create_dataset("nCount_RNA", data=matrix.sum(axis=1) + 1)
        obs.create_dataset("percent.mt", data=[1, 2, 3, 2, 1, 1, 1, 99])
        obs.create_dataset("predicted.celltype.l2.score", data=np.full(8, 0.9))
        var = f.create_group("var")
        var.attrs["_index"] = "_index"
        var.create_dataset("_index", data=["g1", "g2", "g3", "g4"], dtype=string_dtype)
        # Duplicate symbols must remain distinct Ensembl-like primary IDs.
        var.create_dataset("feature_name", data=["A", "B", "A", "D"], dtype=string_dtype)


class PreprocessingTests(unittest.TestCase):
    def test_normalization(self):
        counts = np.array([[1, 3, 0], [2, 0, 2]], dtype=np.int64)
        expression = normalise_counts(counts)
        np.testing.assert_allclose(np.expm1(expression.astype(float)).sum(axis=1), 1e6, rtol=2e-6)
        self.assertEqual(expression.dtype, np.float32)
        self.assertEqual(expression[0, 2], 0)

    def test_qc_exports_png_and_svg(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "qc.png"
            plot_qc(destination, [25, 60, 75], [25, 60], [3, 2, 1], [1, 1, 2, 1], {
                "min_cells": 2, "n_features": [1, 2, 3],
                "mt_percent": [1, 2, 3], "annotation_score": [0.6, 0.8, 1.0],
            })
            self.assertGreater(destination.stat().st_size, 1000)
            self.assertGreater(destination.with_suffix(".svg").stat().st_size, 1000)

    def test_zero_library_rejected(self):
        with self.assertRaises(ValueError):
            normalise_counts(np.zeros((1, 3), dtype=np.int64))

    def test_stream_matches_dense_and_chunk_invariance(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            fixture(base / "input.h5ad")
            arrays = []
            for block in (1, 3, 100):
                summary = preprocess(base / "input.h5ad", base / f"data{block}", base / f"report{block}",
                                     min_cells=2, min_genes=1, max_mt=20,
                                     block_rows=block, plots=False)
                with np.load(base / f"data{block}" / "cd8_pseudobulk.npz", allow_pickle=False) as output:
                    np.testing.assert_array_equal(output["donor_ids"], ["d1", "d2"])
                    np.testing.assert_array_equal(output["counts"], [[4, 2, 2, 1], [2, 4, 1, 0]])
                    self.assertEqual(len(output["gene_ids"]), 4)
                    arrays.append(output["log1p_cpm"].copy())
                self.assertEqual(summary["cells"]["cd8_before_qc"], 7)
                self.assertEqual(summary["cells"]["cd8_after_cell_qc"], 6)
                self.assertEqual(summary["cells"]["cd8_in_retained_donors"], 5)
                self.assertEqual(summary["donors"]["retained"], 2)
                self.assertEqual(summary["features"]["unique_gene_symbols"], 3)
                self.assertEqual(summary["validation"]["raw_integer_count_sum"], 16)
                self.assertEqual(summary["validation"]["umi_source_mismatch_cells"], 6)
                manifest = json.loads((base / f"report{block}" / "summary.json").read_text())
                self.assertEqual(manifest["status"], "completed")
            np.testing.assert_array_equal(arrays[0], arrays[1])
            np.testing.assert_array_equal(arrays[1], arrays[2])
            with self.assertRaises(FileExistsError):
                preprocess(base / "input.h5ad", base / "data1", base / "report1", plots=False)

    def test_fractional_counts_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            fixture(base / "input.h5ad", fractional=True)
            with self.assertRaisesRegex(ValueError, "integer counts"):
                preprocess(base / "input.h5ad", base / "out", base / "report",
                           min_cells=2, min_genes=1, plots=False)
            self.assertFalse((base / "report" / "summary.json").exists())

    def test_conflicting_donor_age_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            fixture(base / "input.h5ad", conflicting_age=True)
            with self.assertRaisesRegex(ValueError, "inconsistent age"):
                preprocess(base / "input.h5ad", base / "out", base / "report",
                           min_cells=2, min_genes=1, plots=False)


if __name__ == "__main__":
    unittest.main()
