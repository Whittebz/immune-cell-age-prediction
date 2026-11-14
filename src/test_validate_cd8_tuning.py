"""Synthetic, non-production tests for the independent tuning validator.

Run: .venv/bin/python -m unittest discover -s src -p test_validate_cd8_tuning.py
No training, source-data loading, or production-result writes occur here.
"""
import copy
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import validate_cd8_tuning as v


def synthetic_plans():
    ids = [f"synthetic-{i:02}" for i in range(24)]
    ages = np.tile([20., 30., 45., 55., 65., 75., 85., 95.], 3)
    pools = np.repeat(np.arange(12).astype(str), 2)
    outer = [(np.arange(12, 24), np.arange(12)), (np.arange(12), np.arange(12, 24))]
    inner = [[(np.arange(6, 12), np.arange(6)), (np.arange(6), np.arange(6, 12))] for _ in outer]
    raw = {p: {"outer": copy.deepcopy(outer), "inner": copy.deepcopy(inner)} for p in v.PROTOCOLS}
    plans = v.validate_split_plan(raw, ids, pools, 2, 2)
    return ids, ages, pools, plans


def synthetic_tuning():
    ids, ages, pools, plans = synthetic_plans()
    config = {"feature_counts": [1, 2], "ridge_alphas": [.1, 1.],
              "elastic_net_alphas": [.01, .1], "l1_ratios": [.1, .9]}
    rows, selected = [], []
    for p in v.PROTOCOLS:
        for o in range(2):
            for m, grid in v.candidate_grid(config).items():
                for i in range(2):
                    for k, alpha, l1 in grid:
                        # Equal means across ALL candidates; macro favors the LAST
                        # candidate, verifying neither macro nor valid_n is used.
                        rows.append(dict(protocol=p, model=m, outer_fold=str(o), inner_fold=str(i),
                                         max_features=k, alpha=alpha, l1_ratio=l1,
                                         mae_years=4. if i == 0 else 6.,
                                         macro_age_mae_years=1. / (k + alpha + (l1 or 0)), train_n=6, valid_n=6))
                first = grid[0]
                selected.append(dict(protocol=p, model=m, outer_fold=str(o), max_features=first[0],
                                     alpha=first[1], l1_ratio=first[2], mean_inner_mae_years=5.))
    return config, plans, rows, selected


def write_table(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metric_fixture(report):
    ids, ages, pools, plans = synthetic_plans()
    oof, aggregates, folds, bands, summaries = [], [], [], [], []
    for p in v.PROTOCOLS:
        baseline = np.empty(len(ids))
        for o, (train, test) in enumerate(plans[p]["outer"]):
            baseline[test] = np.median(ages[train])
        for m in v.MODELS:
            pred = baseline if m == "median" else ages + np.linspace(-3, 5, len(ids)) * (v.MODELS.index(m) + 1)
            pooled = v.regression_metrics(ages, pred, m != "median")
            pooled.update(mae_delta_vs_median_years=float(np.mean(np.abs(pred - ages) - np.abs(baseline - ages))),
                          macro_age_mae_years=v.macro_age_mae(ages, pred))
            aggregates.append(dict(protocol=p, model=m, **pooled))
            fold_parts = []
            for o, (_, test) in enumerate(plans[p]["outer"]):
                fm = v.regression_metrics(ages[test], pred[test], m != "median")
                fm.update(mae_delta_vs_median_years=float(np.mean(np.abs(pred[test] - ages[test]) - np.abs(baseline[test] - ages[test]))),
                          macro_age_mae_years=v.macro_age_mae(ages[test], pred[test]))
                fold_parts.append(fm)
                folds.append(dict(protocol=p, model=m, fold=o, **fm))
                for j in test:
                    oof.append(dict(protocol=p, model=m, donor_id=ids[j], fold=o, actual_age=ages[j],
                                    predicted_age=pred[j], raw_age_gap=pred[j] - ages[j]))
            summaries.append(dict(protocol=p, model=m, **v.fold_summary(pooled, fold_parts)))
            for k, label in enumerate(v.AGE_LABELS):
                mask = v.age_indices(ages) == k
                bm = v.regression_metrics(ages[mask], pred[mask], False)
                bm.pop("spearman")
                bands.append(dict(protocol=p, model=m, age_band=label, **bm,
                                  signed_residual_mean=float(np.mean(pred[mask] - ages[mask]))))
    for name, rows in (("metrics.csv", aggregates), ("fold_metrics.csv", folds),
                       ("age_band_metrics.csv", bands), ("fold_summary.csv", summaries)):
        write_table(report / name, rows)
    return ids, ages, plans, oof


class MetricHelpersTest(unittest.TestCase):
    def test_tied_spearman_is_rank_not_pearson(self):
        result = v.regression_metrics([20, 20, 50, 90], [2, 1, 1, 9])
        # Average ranks: (1.5,1.5,3,4) vs (3,1.5,1.5,4).
        a, b = np.array([1.5, 1.5, 3, 4]), np.array([3, 1.5, 1.5, 4])
        self.assertAlmostEqual(result["spearman"], np.corrcoef(a, b)[0, 1])
        self.assertNotAlmostEqual(result["spearman"], np.corrcoef([20, 20, 50, 90], [2, 1, 1, 9])[0, 1])

    def test_median_rank_is_always_none_even_varying_predictions(self):
        self.assertIsNone(v.regression_metrics([20, 50, 90], [30, 60, 65], False)["spearman"])
        self.assertIsNone(v.regression_metrics([20, 50, 90], [50, 50, 50])["spearman"])

    def test_age_edges_and_observed_macro(self):
        np.testing.assert_array_equal(v.age_indices([18, 39, 40, 59, 60, 79, 80, 100]), [0, 0, 1, 1, 2, 2, 3, 3])
        # One error in sparse young bin, two zero errors in older bin.
        self.assertEqual(v.macro_age_mae([20, 65, 70], [30, 65, 70]), 5.)
        for age in (17.9, 101, np.nan):
            with self.assertRaises(ValueError):
                v.age_indices([age])

    def test_regression_and_invalid_inputs(self):
        metrics = v.regression_metrics([20, 40], [22, 44])
        self.assertEqual(metrics["mae_years"], 3)
        self.assertAlmostEqual(metrics["rmse_years"], np.sqrt(10))
        self.assertAlmostEqual(metrics["r2"], .9)
        self.assertEqual(v.regression_metrics([20, 20], [20, 20])["r2"], 1)
        self.assertEqual(v.regression_metrics([20, 20], [21, 20])["r2"], 0)
        self.assertIsNone(v.regression_metrics([20], [21])["r2"])
        with self.assertRaises(ValueError):
            v.regression_metrics([20], [np.inf])

    def test_sample_sd_and_weighted_mae(self):
        folds = [dict(n=1, mae_years=3., rmse_years=4., r2=.1),
                 dict(n=3, mae_years=5., rmse_years=6., r2=.5)]
        result = v.fold_summary(dict(n=4, mae_years=4.5, rmse_years=5.6, r2=.4), folds)
        self.assertEqual(result["unweighted_fold_mean_mae_years"], 4)
        self.assertAlmostEqual(result["fold_sample_sd_ddof1_mae_years"], np.sqrt(2))
        self.assertEqual(result["n_weighted_fold_mae_years"], 4.5)
        self.assertTrue(result["weighted_mae_matches_pooled"])
        with self.assertRaises(ValueError):
            v.fold_summary(dict(n=99), folds)

    def test_key_coverage_and_duplicates(self):
        for rows in ([{"k": "a"}, {"k": "a"}], [{"k": "a"}]):
            with self.assertRaises(ValueError):
                v.keyed(rows, ("k",), {("a",), ("b",)})


class SelectionTest(unittest.TestCase):
    def setUp(self):
        self.config, self.plans, self.rows, self.selected = synthetic_tuning()

    def test_planned_full_grid_counts(self):
        config = dict(feature_counts=[100, 300, 1000], ridge_alphas=[.1, 1, 10, 100, 300, 1000],
                      elastic_net_alphas=[.01, .1, 1, 3, 10], l1_ratios=[.1, .5, .9])
        grid = v.candidate_grid(config)
        self.assertEqual(len(grid["ridge_expanded"]), 18)
        self.assertEqual(len(grid["en_expanded"]), 45)
        self.assertEqual(len(grid["en_age_weighted"]), 45)
        self.assertEqual(sum(map(len, grid.values())) * 2 * 5 * 3, 3240)

    def test_invalid_config_grid_fails(self):
        for key, value in (("feature_counts", [1.5]), ("ridge_alphas", [.1, .1]),
                           ("elastic_net_alphas", [1., .01]), ("l1_ratios", [1.1])):
            config = copy.deepcopy(self.config)
            config[key] = value
            with self.assertRaises(ValueError):
                v.candidate_grid(config)

    def test_coverage_and_deterministic_tie_mae_only(self):
        scores, winners = v.validate_tuning(self.rows, self.selected, self.config, self.plans)
        self.assertEqual(len(scores), 160)
        for (p, model, outer), candidate in winners.items():
            self.assertEqual(candidate, v.candidate_grid(self.config)[model][0])
        # Input order is immaterial for deterministic tie breaking.
        v.validate_tuning(self.rows[::-1], self.selected[::-1], self.config, self.plans)

    def test_missing_duplicate_unplanned_candidates_fail(self):
        variants = [self.rows[:-1], self.rows + [self.rows[0]], copy.deepcopy(self.rows)]
        variants[-1][0]["alpha"] = 999
        for rows in variants:
            with self.assertRaises(ValueError):
                v.validate_tuning(rows, self.selected, self.config, self.plans)

    def test_selection_wrong_mean_or_macro_choice_fails(self):
        for field, value in (("mean_inner_mae_years", 4.), ("max_features", 2)):
            selected = copy.deepcopy(self.selected)
            selected[0][field] = value
            with self.assertRaises(ValueError):
                v.validate_tuning(self.rows, selected, self.config, self.plans)

    def test_train_size_and_ridge_l1_are_checked(self):
        for field, value in (("train_n", 99), ("l1_ratio", .1)):
            rows = copy.deepcopy(self.rows)
            rows[0][field] = value
            with self.assertRaises(ValueError):
                v.validate_tuning(rows, self.selected, self.config, self.plans)

    def test_selection_uses_unweighted_fold_mean(self):
        # Deliberately unequal inner validation sizes: weighted means would choose
        # alpha=1, but unweighted means choose alpha=.1.
        rows, plans, selected = copy.deepcopy(self.rows), copy.deepcopy(self.plans), copy.deepcopy(self.selected)
        for p in v.PROTOCOLS:
            for o in range(2):
                plans[p]["inner"][o] = [(np.arange(10), np.arange(10, 12)), (np.arange(2), np.arange(2, 12))]
        for row in rows:
            i = int(row["inner_fold"])
            row["train_n"], row["valid_n"] = (10, 2) if i == 0 else (2, 10)
            row["mae_years"] = (0 if i == 0 else 6) if row["alpha"] in (.1, .01) else (7 if i == 0 else 0)
        for row in selected:
            row["mean_inner_mae_years"] = 3.
        v.validate_tuning(rows, selected, self.config, plans)

    def test_diagnostics_cover_all_fits_and_preserve_warning_counts(self):
        scores, winners = v.validate_tuning(self.rows, self.selected, self.config, self.plans)
        records = []
        for (p, m, o, i, k, alpha, l1), _ in scores.items():
            records.append(dict(protocol=p, model=m, outer_fold=o, inner_fold=i, phase="inner_tuning",
                                max_features=k, alpha=alpha, l1_ratio=l1, train_n=6,
                                n_iter=None if m == "ridge_expanded" else 10,
                                dual_gap=None if m == "ridge_expanded" else .00001,
                                convergence_warning_count=0, warnings=[]))
        for (p, m, o), (k, alpha, l1) in winners.items():
            records.append(dict(protocol=p, model=m, outer_fold=o, inner_fold=-1, phase="outer_refit",
                                max_features=k, alpha=alpha, l1_ratio=l1, train_n=12,
                                n_iter=None if m == "ridge_expanded" else 10,
                                dual_gap=None if m == "ridge_expanded" else .00001,
                                convergence_warning_count=0, warnings=[]))
        records[-1]["warnings"] = [{"category": "ConvergenceWarning", "message": "synthetic warning"},
                                   {"category": "UserWarning", "message": "also preserved"}]
        records[-1]["convergence_warning_count"] = 1
        self.assertEqual(v.validate_diagnostics(records, scores, winners, self.plans, 100), (1, 1))
        for changed in (records[:-1], records + [records[0]]):
            with self.assertRaises(ValueError):
                v.validate_diagnostics(changed, scores, winners, self.plans, 100)
        records[-1]["convergence_warning_count"] = 0
        with self.assertRaises(ValueError):
            v.validate_diagnostics(records, scores, winners, self.plans, 100)


class EarlyZeroSolutionDiagnosticsTest(unittest.TestCase):
    def test_zero_iteration_elastic_net_early_solution_is_valid(self):
        # Saved-diagnostic regression fixture only: no estimator is trained.
        # sklearn 1.9.1 can accept an all-zero high-penalty solution before
        # any coordinate sweep; its reported dual gap need not be exactly zero.
        _, _, _, plans = synthetic_plans()
        for model in ("en_expanded", "en_age_weighted"):
            key = ("donor_cv", model, 0, 0, 100, 10., .9)
            scores = {key: 12.}
            row = dict(protocol="donor_cv", model=model, outer_fold=0, inner_fold=0,
                       phase="inner_tuning", max_features=100, alpha=10., l1_ratio=.9,
                       train_n=6, n_iter=0, dual_gap=0., convergence_warning_count=0, warnings=[])
            for gap in (0., .007247655):
                row["dual_gap"] = gap
                self.assertEqual(v.validate_diagnostics([row], scores, {}, plans, 100), (0, 0))
            row["n_iter"] = 100
            self.assertEqual(v.validate_diagnostics([row], scores, {}, plans, 100), (0, 0))
            for invalid in (-1, 101):
                row["n_iter"] = invalid
                with self.assertRaises(ValueError):
                    v.validate_diagnostics([row], scores, {}, plans, 100)


class WeightAndSplitTest(unittest.TestCase):
    def test_weights_are_training_only_normalized_binary_hashed(self):
        ids, ages, _, plans = synthetic_plans()
        rows = []
        for p in v.PROTOCOLS:
            for o, (outer_train, test) in enumerate(plans[p]["outer"]):
                for i in [-1, 0, 1]:
                    train, valid = outer_train, test
                    if i >= 0:
                        a, b = plans[p]["inner"][o][i]
                        train, valid = outer_train[a], outer_train[b]
                    rows.append(dict(protocol=p, outer_fold=str(o), inner_fold=str(i),
                                     **v.weight_audit(ids, ages, train, valid)))
        v.validate_weight_audits(rows, ids, ages, plans)
        weights, counts = v.age_weights([20, 20, 25, 65])
        expected = np.array([3 ** -.5] * 3 + [1.])
        expected /= expected.mean()
        np.testing.assert_array_equal(weights, expected)
        self.assertEqual(counts.tolist(), [3, 0, 1, 0])
        audit = v.weight_audit(["b", "a", "d", "c"], [20, 20, 25, 65], [2, 0, 3], [1])
        self.assertEqual(audit["train_ids_sha256"], v.canonical_digest(["d", "b", "c"]))
        actual_weights, _ = v.age_weights([25, 20, 65])
        self.assertEqual(audit["weights_sha256"], hashlib.sha256(actual_weights.astype("<f8").tobytes()).hexdigest())
        for field, value in (("weight_mean", 2), ("weights_sha256", "wrong"), ("train_ids_sha256", "sorted-ids")):
            changed = copy.deepcopy(rows)
            changed[0][field] = value
            with self.assertRaises(ValueError):
                v.validate_weight_audits(changed, ids, ages, plans)

    def test_old_demo_plan_is_present_but_unused(self):
        ids, _, pools, plans = synthetic_plans()
        plans["demo_inner"] = ["old demo plan is deliberately not consumed"]
        v.validate_split_plan(plans, ids, pools, 2, 2)
        plans["unknown_strategy"] = []
        with self.assertRaises(ValueError):
            v.validate_split_plan(plans, ids, pools, 2, 2)

    def test_relative_inner_and_pool_leakage_failures(self):
        ids, _, pools, plans = synthetic_plans()
        for mutate in ("absolute-inner", "pool-leak", "missing-test"):
            changed = copy.deepcopy(plans)
            if mutate == "absolute-inner":
                changed["donor_cv"]["inner"][0][0] = (np.arange(18, 24), np.arange(12, 18))
            elif mutate == "pool-leak":
                train, test = changed["pool_cv"]["outer"][0]
                train[0], test[0] = test[0], train[0]
            else:
                changed["donor_cv"]["outer"][1] = changed["donor_cv"]["outer"][0]
            with self.assertRaises(ValueError):
                v.validate_split_plan(changed, ids, pools, 2, 2)


class SavedMetricTablesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="cd8-validator-synthetic-")
        self.addCleanup(self.tmp.cleanup)
        self.report = Path(self.tmp.name)
        self.ids, self.ages, self.plans, self.oof = metric_fixture(self.report)

    def check_tables(self, rows=None):
        return v.validate_metric_tables(self.oof if rows is None else rows, self.ids, self.ages, self.plans, self.report)

    def test_complete_tables(self):
        predictions, metrics = self.check_tables()
        self.assertEqual(len(predictions), 12)
        self.assertEqual(len(metrics), 12)

    def test_wrong_sample_sd_macro_and_rank_fail(self):
        for name, field, model in (("fold_summary.csv", "fold_sample_sd_ddof1_mae_years", "ridge_v1"),
                                   ("metrics.csv", "macro_age_mae_years", "ridge_v1"),
                                   ("metrics.csv", "spearman", "median")):
            with self.subTest(field=field):
                metric_fixture(self.report)
                rows = v.read_csv(self.report / name)
                row = next(row for row in rows if row["model"] == model)
                row[field] = 123
                write_table(self.report / name, rows)
                with self.assertRaises(ValueError):
                    self.check_tables()

    def test_raw_gap_actual_age_fold_and_coverage_fail(self):
        for field, value in (("actual_age", 99), ("fold", 88), ("raw_age_gap", 999), ("predicted_age", "nan")):
            rows = copy.deepcopy(self.oof)
            rows[0][field] = value
            with self.assertRaises(ValueError):
                self.check_tables(rows)
        with self.assertRaises(ValueError):
            self.check_tables(self.oof[:-1])
        with self.assertRaises(ValueError):
            self.check_tables(self.oof + [self.oof[0]])

    def test_missing_metric_group_and_age_bias_fail(self):
        rows = v.read_csv(self.report / "metrics.csv")
        write_table(self.report / "metrics.csv", rows[:-1])
        with self.assertRaises(ValueError):
            self.check_tables()
        metric_fixture(self.report)
        rows = v.read_csv(self.report / "age_band_metrics.csv")
        rows[0]["signed_residual_mean"] = 999
        write_table(self.report / "age_band_metrics.csv", rows)
        with self.assertRaises(ValueError):
            self.check_tables()


class ProvenanceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="cd8-provenance-synthetic-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.shared, self.private = self.root / "data/shared", self.root / "private"
        self.private.mkdir()
        base_path = self.root / "config/base.json"
        self.addCleanup(patch.stopall)
        patch.object(v, "ROOT", self.root).start()
        patch.object(v, "DEFAULT_CONFIG", base_path).start()
        core = [self.root / "data/models/CD8/common_v1" / name for name in
                ("analysis_cohort.csv", "fold_assignments.csv", "oof_predictions.csv")]
        core += [self.root / "data/models/CD8/pool_en_common_v1/oof_predictions.csv", base_path]
        core += [self.shared / name for name in ("protocol.json", "common_cohort.csv", "splits.json")]
        for path in core + [self.root / "src/tune_cd8_common.py", self.root / "data/processed/CD8/cd8_pseudobulk.npz"]:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("synthetic-only fixture\n", encoding="utf-8")
        base = {"base": "original configuration"}
        base_path.write_text(json.dumps(base, indent=4) + "\n", encoding="utf-8")
        self.frozen = {"cohort_sha256": "frozen-cohort", "splits_sha256": "frozen-splits",
                       "config_sha256": v.canonical_digest(base)}
        config = {"synthetic_grid": [1, 2]}
        source = {"cohort_sha256": "frozen-cohort", "splits_sha256": "frozen-splits",
                  "config_canonical_sha256": v.canonical_digest(base),
                  "bundle_sha256": v.sha256(self.root / "data/processed/CD8/cd8_pseudobulk.npz")}
        self.manifest = dict(config=config, config_sha256=v.canonical_digest(config),
                             code_sha256=v.sha256(self.root / "src/tune_cd8_common.py"), n_donors=2, n_input_genes=3,
                             protected_inputs={str(path): v.sha256(path) for path in core}, source=source)
        self.summary = dict(configuration=config, source=source, n_donors=2, n_input_genes=3, private_artifacts={})
        for name in ("analysis_cohort.csv", "fold_assignments.csv", "oof_predictions.csv", "inner_tuning_scores.csv",
                     "age_weight_audit.csv", "fit_diagnostics.json", "outer_models.npz", "run_manifest.json"):
            path = self.private / name
            path.write_text("synthetic artifact", encoding="utf-8")
            self.summary["private_artifacts"][name] = dict(sha256=v.sha256(path), bytes=path.stat().st_size)

    def validate(self):
        v.validate_provenance(self.manifest, self.summary, self.private, self.shared, self.frozen, 2, 3)

    def test_canonical_original_config_digest_not_raw_bytes(self):
        self.assertNotEqual(self.frozen["config_sha256"], v.sha256(v.DEFAULT_CONFIG))
        self.validate()
        self.manifest["source"]["config_canonical_sha256"] = v.sha256(v.DEFAULT_CONFIG)
        with self.assertRaises(ValueError):
            self.validate()

    def test_tuning_config_digest_and_code_seal(self):
        self.validate()
        self.manifest["config_sha256"] = "incorrect raw-file hash"
        with self.assertRaises(ValueError):
            self.validate()
        self.manifest["config_sha256"] = v.canonical_digest(self.manifest["config"])
        self.manifest["code_sha256"] = "wrong code hash"
        with self.assertRaises(ValueError):
            self.validate()

    def test_changed_protected_input_and_missing_core_fail(self):
        path = next(iter(self.manifest["protected_inputs"]))
        digest = self.manifest["protected_inputs"].pop(path)
        with self.assertRaises(ValueError):
            self.validate()
        self.manifest["protected_inputs"][path] = digest
        Path(path).write_text("mutated synthetic input", encoding="utf-8")
        with self.assertRaises(ValueError):
            self.validate()

    def test_missing_or_changed_private_artifact_fails(self):
        self.summary["private_artifacts"]["outer_models.npz"]["bytes"] += 1
        with self.assertRaises(ValueError):
            self.validate()
        self.summary["private_artifacts"].pop("outer_models.npz")
        with self.assertRaises(ValueError):
            self.validate()


class NumericArchiveTest(unittest.TestCase):
    def test_source_reconstruction_and_unselected_defaults(self):
        ids, ages, pools, plans = synthetic_plans()
        X = np.column_stack([np.arange(24) * .1, np.arange(24) ** 2, np.arange(24) * 10., np.ones(24)])
        genes = ["synthetic-gene-" + str(i) for i in range(4)]
        labels = [p + "/" + m for p in v.PROTOCOLS for m in v.NEW_MODELS]
        coefficients = np.zeros((6, 2, 4))
        available = np.zeros((6, 2, 4), bool)
        means, scales, intercepts = coefficients.copy(), np.ones_like(coefficients), np.ones((6, 2))
        winners, predictions = {}, {}
        for g, label in enumerate(labels):
            p, model = label.split("/")
            pred = np.empty(24)
            for o, (train, test) in enumerate(plans[p]["outer"]):
                winners[p, model, o] = (2, 1., None if model == "ridge_expanded" else .5)
                selected = np.array([1, 2])
                available[g, o, selected] = True
                means[g, o, selected] = X[train][:, selected].mean(axis=0)
                scales[g, o, selected] = X[train][:, selected].std(axis=0)
                coefficients[g, o, selected] = [.1, .2]
                pred[test] = ((X[test][:, selected] - means[g, o, selected]) / scales[g, o, selected]) @ coefficients[g, o, selected] + 1
            predictions[p, model] = pred
        with tempfile.TemporaryDirectory(prefix="cd8-model-synthetic-") as directory:
            path = Path(directory) / "synthetic.npz"
            def save():
                np.savez(path, group_labels=np.array(labels), gene_ids=np.array(genes), coefficients=coefficients,
                         available=available, means=means, scales=scales, intercepts=intercepts)
            save()
            v.validate_outer_archive(path, plans, winners, genes, predictions, ids, X)
            coefficients[0, 0, 0] = 1
            save()
            with self.assertRaises(ValueError):
                v.validate_outer_archive(path, plans, winners, genes, predictions, ids, X)
            coefficients[0, 0, 0] = 0
            intercepts[0, 0] += 1
            save()
            with self.assertRaises(ValueError):
                v.validate_outer_archive(path, plans, winners, genes, predictions, ids, X)


if __name__ == "__main__":
    unittest.main()
