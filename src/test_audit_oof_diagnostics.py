"""In-memory unit checks; no participant-level outputs or source writes."""
import copy
import unittest
from unittest.mock import patch
from pathlib import Path
from argparse import Namespace
import io

import numpy as np

from audit_oof_diagnostics import (
    cell_band, compare_source_metrics, compute_tables, depth_quartiles,
    nuisance_design, rank_association, regression_metrics, validate_and_join,
    select_analysis_metadata, prepare_output, write_csv, read_csv, ROOT, run,
)


def fixture():
    metadata, assignments, base = [], [], []
    for i in range(10):
        donor, age, fold = f"synthetic_{i}", 20 + i * 5, i % 5
        metadata.append({"donor_id": donor, "age": str(age), "sex": "male" if i % 2 else "female",
                         "pool_number": str(fold), "n_cd8_cells": str(30 + i * 35),
                         "actual_cd8_umi": str(100000 + i * 10000), "included": "1"})
        assignments.append({"protocol": "pool_cv", "donor_id": donor, "fold": str(fold), "pool_number": str(fold)})
        base.append({"protocol": "pool_cv", "model": "median", "donor_id": donor,
                     "fold": str(fold), "actual_age": str(age), "predicted_age": "45", "raw_age_gap": str(45-age)})
    return metadata, assignments, base


class OOFDiagnosticsTests(unittest.TestCase):
    def test_metrics_and_constant_age_r2(self):
        result = regression_metrics([1, 2, 3], [2, 2, 2])
        self.assertAlmostEqual(result["mae_years"], 2 / 3)
        self.assertAlmostEqual(result["rmse_years"], np.sqrt(2 / 3))
        self.assertEqual(result["r2"], 0)
        self.assertTrue(np.isnan(regression_metrics([2, 2], [2, 2])["r2"]))

    def test_cell_boundaries(self):
        self.assertEqual([cell_band(x) for x in [29, 30, 59, 60, 119, 120, 239, 240]],
                         ["<30", "30-59", "30-59", "60-119", "60-119", "120-239", "120-239", ">=240"])

    def test_quartile_ties_not_split(self):
        labels, cuts = depth_quartiles([1, 2, 2, 2, 3, 4, 5, 6])
        self.assertEqual(len(set(labels[1:4])), 1)
        self.assertEqual(len(cuts), 3)

    def test_full_cohort_supplement(self):
        metadata, folds, base = fixture()
        extra = [{**r, "model": "elastic_net"} for r in base]
        groups, keys = validate_and_join([base, extra], metadata, folds)
        self.assertEqual(len(groups), 2)
        self.assertEqual(keys, {("pool_cv", "median")})

    def test_duplicate_supplement_rejected(self):
        metadata, folds, base = fixture()
        with self.assertRaisesRegex(ValueError, "Duplicate protocol/model/donor"):
            validate_and_join([base, base], metadata, folds)

    def test_partial_cohort_rejected(self):
        metadata, folds, base = fixture()
        extra = [{**r, "model": "elastic_net"} for r in base[:-1]]
        with self.assertRaisesRegex(ValueError, "complete included metadata cohort"):
            validate_and_join([base, extra], metadata, folds)

    def test_age_fold_and_residual_mismatch_rejected(self):
        for field, value, message in [("actual_age", "18", "actual age"),
                                      ("fold", "4", "fold/pool"),
                                      ("raw_age_gap", "99", "signed residual")]:
            with self.subTest(field=field):
                metadata, folds, base = fixture()
                base[0][field] = value
                with self.assertRaisesRegex(ValueError, message):
                    validate_and_join([base], metadata, folds)

    def test_pool_leakage_rejected(self):
        metadata, folds, base = fixture()
        metadata[0]["pool_number"] = metadata[1]["pool_number"]
        folds[0]["pool_number"] = folds[1]["pool_number"]
        with self.assertRaisesRegex(ValueError, "multiple pool-CV"):
            validate_and_join([base], metadata, folds)

    def test_fold_summary_ddof1_and_weighted_identity(self):
        metadata, folds, base = fixture()
        groups, _ = validate_and_join([base], metadata, folds)
        tables, _ = compute_tables(groups)
        summary = tables["fold_summary"][0]
        mae = [r["mae_years"] for r in tables["fold_metrics_recomputed"]]
        self.assertAlmostEqual(summary["fold_sample_sd_ddof1_mae_years"], np.std(mae, ddof=1))
        self.assertAlmostEqual(summary["unweighted_fold_mean_mae_years"], np.mean(mae))
        self.assertTrue(summary["weighted_mae_matches_pooled"])
        self.assertNotAlmostEqual(summary["pooled_r2"], summary["unweighted_fold_mean_r2"])
        self.assertFalse(any("donor_id" in row for rows in tables.values() for row in rows))

    def test_unequal_fold_weights_are_not_unweighted_mean(self):
        metadata, folds, base = fixture()
        metadata.append({**metadata[0], "donor_id": "synthetic_extra", "age": "90"})
        folds.append({**folds[0], "donor_id": "synthetic_extra"})
        base.append({**base[0], "donor_id": "synthetic_extra", "actual_age": "90", "raw_age_gap": "-45"})
        groups, _ = validate_and_join([base], metadata, folds)
        tables, _ = compute_tables(groups)
        summary = tables["fold_summary"][0]
        self.assertTrue(summary["weighted_mae_matches_pooled"])
        self.assertNotAlmostEqual(summary["pooled_mae_years"], summary["unweighted_fold_mean_mae_years"])

    def test_source_tolerance_and_corruption_detection(self):
        metadata, folds, base = fixture()
        groups, keys = validate_and_join([base], metadata, folds)
        tables, _ = compute_tables(groups)
        pooled, fold_rows = tables["pooled_metrics"], tables["fold_metrics_recomputed"]
        checks = compare_source_metrics(pooled, fold_rows, copy.deepcopy(pooled), copy.deepcopy(fold_rows), keys)
        self.assertTrue(all(r["passed"] for r in checks))
        corrupted = copy.deepcopy(pooled)
        corrupted[0]["mae_years"] += .1
        checks = compare_source_metrics(pooled, fold_rows, corrupted, fold_rows, keys)
        self.assertFalse(all(r["passed"] for r in checks))
        with self.assertRaisesRegex(ValueError, "missing source metric"):
            compare_source_metrics(pooled, fold_rows, [], fold_rows, keys)

    def test_rank_diagnostic_matches_explicit_projection(self):
        rng = np.random.default_rng(5)
        age = np.linspace(20, 90, 100)
        sex = np.where(np.arange(100) % 2, "male", "female")
        pool = np.asarray([str(i % 5) for i in range(100)])
        design = nuisance_design(age, sex, pool)
        x = age + rng.normal(size=100) * 5
        y = (age - 55) ** 2 + rng.normal(size=100) * 20
        result = rank_association(x, y, design)
        from scipy.stats import rankdata
        a, b = rankdata(x), rankdata(y)
        a -= design @ np.linalg.lstsq(design, a, rcond=None)[0]
        b -= design @ np.linalg.lstsq(design, b, rcond=None)[0]
        self.assertAlmostEqual(result["adjusted_rank_residual_r"], np.corrcoef(a, b)[0, 1])
        self.assertEqual(result["n"], 100)
        self.assertGreater(result["residual_df"], 3)

    def test_declared_common_subset_and_no_old_oof_slicing(self):
        metadata, folds, base = fixture()
        cohort = copy.deepcopy(metadata[:5])
        selected = select_analysis_metadata(metadata, cohort, {"n_donors": 5})
        groups, _ = validate_and_join([base[:5]], selected, folds)
        self.assertEqual(len(groups[("pool_cv", "median")]), 5)
        with self.assertRaisesRegex(ValueError, "unknown/excluded"):
            validate_and_join([base], selected, folds)

    def test_declared_metadata_guards(self):
        metadata, _, _ = fixture()
        for field, value in [("age", "20.00001"), ("sex", "male"), ("pool_number", "wrong"),
                             ("n_cd8_cells", "31"), ("actual_cd8_umi", "100001")]:
            with self.subTest(field=field):
                cohort = copy.deepcopy(metadata[:5])
                cohort[0][field] = value
                with self.assertRaisesRegex(ValueError, "differs from preparation source"):
                    select_analysis_metadata(metadata, cohort, {"n_donors": 5})
        with self.assertRaisesRegex(ValueError, "n_donors"):
            select_analysis_metadata(metadata, metadata[:5], {"n_donors": 10})
        with self.assertRaisesRegex(ValueError, "requires training/summary"):
            select_analysis_metadata(metadata, metadata[:5])
        with self.assertRaisesRegex(ValueError, "unique subset"):
            select_analysis_metadata(metadata, metadata[:5] + metadata[:1], {"n_donors": 6})
        unknown = copy.deepcopy(metadata[:5])
        unknown[0]["donor_id"] = "not_in_source"
        with self.assertRaisesRegex(ValueError, "unique subset"):
            select_analysis_metadata(metadata, unknown, {"n_donors": 5})

    def test_dynamic_b_lineage_fields(self):
        metadata, folds, base = fixture()
        for row in metadata:
            row["n_b_cells"] = row.pop("n_cd8_cells")
            row["actual_b_umi"] = row.pop("actual_cd8_umi")
        selected = select_analysis_metadata(metadata, metadata[:5], {"n_donors": 5}, cell="B")
        groups, _ = validate_and_join([base[:5]], selected, folds, cell="B")
        tables, _ = compute_tables(groups, cell="B")
        self.assertEqual({r["exposure"] for r in tables["error_depth_associations"]}, {"n_b_cells", "actual_b_umi"})
        with self.assertRaisesRegex(ValueError, "schema missing"):
            validate_and_join([base[:5]], selected, folds)

    def test_nonempty_output_refusal_without_writes(self):
        with patch.object(Path, "exists", return_value=True), patch.object(Path, "is_dir", return_value=True), \
             patch.object(Path, "iterdir", return_value=iter([Path("marker")])), patch.object(Path, "mkdir") as mkdir:
            with self.assertRaisesRegex(ValueError, "nonempty output"):
                prepare_output("synthetic_output")
            mkdir.assert_not_called()

    def test_csv_uses_lf(self):
        stream = io.StringIO(newline="")
        with patch.object(Path, "open") as opened:
            opened.return_value.__enter__.return_value = stream
            write_csv("synthetic_output.csv", [{"count": 5, "mae": 1.2}])
        self.assertEqual(stream.getvalue(), "count,mae\n5,1.2\n")

    def test_run_hashes_declared_cohort_and_summary_not_as_extra_oof(self):
        metadata, assignments, base = fixture()
        cohort = metadata[:5]
        groups, _ = validate_and_join([base[:5]], cohort, assignments)
        source, _ = compute_tables(groups)
        data, models, training, output = (Path("/synthetic") / s for s in ("data", "models", "training", "output"))
        extra = Path("/synthetic/extra_oof.csv")
        rows = {data / "donor_metadata.csv": metadata, models / "analysis_cohort.csv": cohort,
                models / "fold_assignments.csv": assignments, models / "oof_predictions.csv": base[:5],
                training / "metrics.csv": source["pooled_metrics"], training / "fold_metrics.csv": source["fold_metrics_recomputed"],
                extra: [{**r, "model": "elastic_net"} for r in base[:5]]}
        args = Namespace(data=data, models=models, training=training, output=output, cell="CD8", extra_oof=[extra])
        with patch("audit_oof_diagnostics.prepare_output", return_value=output), \
             patch("audit_oof_diagnostics.resolve_dir", side_effect=lambda p, *a, **k: Path(p)), \
             patch("audit_oof_diagnostics.sha256", return_value="synthetic_sha256"), \
             patch("audit_oof_diagnostics.read_csv", side_effect=lambda p: rows[Path(p)]), \
             patch.object(Path, "is_file", return_value=True), \
             patch.object(Path, "read_text", return_value='{"n_donors": 5}'), \
             patch.object(Path, "mkdir"), patch.object(Path, "write_text"), \
             patch("audit_oof_diagnostics.write_csv"), patch("audit_oof_diagnostics.make_plots"), patch("builtins.print"):
            status = run(args)
        self.assertEqual(status["cohort_n"], 5)
        self.assertEqual(status["protocol_model_groups"], 2)
        self.assertIn(str(models / "analysis_cohort.csv"), status["source_files_sha256"])
        self.assertIn(str(training / "summary.json"), status["source_files_sha256"])
        self.assertTrue(status["training_summary_n_donors_verified"])

    def test_current_cd8_frozen_metrics_unchanged(self):
        models = ROOT / "data/models/CD8"
        if not (models / "oof_predictions.csv").is_file():
            self.skipTest("Project frozen CD8 data not available")
        training = ROOT / "results/cd8_training"
        metadata = select_analysis_metadata(read_csv(ROOT / "data/processed/CD8/donor_metadata.csv"))
        groups, keys = validate_and_join([read_csv(models / "oof_predictions.csv")], metadata,
                                         read_csv(models / "fold_assignments.csv"))
        tables, _ = compute_tables(groups)
        checks = compare_source_metrics(tables["pooled_metrics"], tables["fold_metrics_recomputed"],
                                         read_csv(training / "metrics.csv"), read_csv(training / "fold_metrics.csv"), keys)
        self.assertEqual(len(checks), 143)
        self.assertTrue(all(row["passed"] for row in checks))

    def test_constant_rank_diagnostic(self):
        design = np.ones((20, 1))
        result = rank_association(np.ones(20), np.arange(20), design)
        self.assertTrue(np.isnan(result["raw_spearman"]))
        self.assertTrue(np.isnan(result["adjusted_rank_residual_r"]))


if __name__ == "__main__":
    unittest.main()
