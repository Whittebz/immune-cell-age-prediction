"""Synthetic unit/integration tests; not empirical evidence of immune aging."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from validate_cd8_training import validate as validate_training
from train_cd8 import (
    TopVarianceGenes, feature_stability, fit_recorded, make_splits,
    metrics, predict_numeric_model, run_benchmark, save_numeric_model, tune_models,
)


class TrainingTests(unittest.TestCase):
    def test_selector_fits_training_rows_only_and_excludes_constant_genes(self):
        train = np.array([[0, 1, 3], [1, 1, 2], [2, 1, 1]], dtype=float)
        test = np.array([[1, 10000, 2]], dtype=float)
        selector = TopVarianceGenes(1).fit(train)
        np.testing.assert_array_equal(selector.indices_, [0])
        np.testing.assert_array_equal(selector.transform(test), [[1]])
        with self.assertRaises(ValueError):
            TopVarianceGenes(2).fit(np.ones((4, 3)))

    def test_grouped_splits_are_disjoint_and_complete(self):
        X = np.arange(120).reshape(30, 4)
        y = np.arange(30)
        pools = np.repeat(np.arange(6), 5)
        splits = make_splits(X, y, pools, 3, 42, True)
        for train, test in splits:
            self.assertFalse(set(pools[train]) & set(pools[test]))
        self.assertEqual(sorted(np.concatenate([t for _, t in splits]).tolist()), list(range(30)))

    def test_inner_preprocessing_refits_for_every_inner_training_split(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(18, 6))
        y = rng.normal(size=18)
        splits = make_splits(X, y, np.arange(18), 3, 42)
        observations = []
        original_fit = TopVarianceGenes.fit
        def audited_fit(selector, data, target=None):
            observations.append(data.copy())
            return original_fit(selector, data, target)
        with patch.object(TopVarianceGenes, 'fit', new=audited_fit):
            best = tune_models(X, y, splits, ['ridge'], 3, [], {'protocol': 'test'}, tuning_rows=[])
        self.assertEqual(len(observations), 3)
        for observed, (train, _) in zip(observations, splits):
            np.testing.assert_array_equal(observed, X[train])
        self.assertIn(best['ridge']['alpha'], [0.1, 1, 10, 100])

    def test_numeric_model_roundtrip_and_feature_schema_check(self):
        rng = np.random.default_rng(2)
        X = rng.normal(size=(30, 6)); y = X[:, 0] * 3 + 50
        genes = np.array([f'g{i}' for i in range(6)])
        pipeline = Pipeline([('genes', TopVarianceGenes(5)), ('scale', StandardScaler()),
                             ('regressor', ElasticNet(alpha=.1, l1_ratio=.5))]).fit(X, y)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'model.npz'
            save_numeric_model(path, pipeline, genes)
            np.testing.assert_allclose(predict_numeric_model(X, genes, path), pipeline.predict(X), atol=1e-10)
            with self.assertRaises(ValueError):
                predict_numeric_model(X, genes[::-1], path)

    def test_nonconvergence_is_recorded(self):
        rng = np.random.default_rng(3)
        X = rng.normal(size=(30, 40)); y = rng.normal(size=30)
        diagnostics = []
        fit_recorded(ElasticNet(alpha=1e-9, max_iter=1, tol=1e-15), X, y, diagnostics,
                     {'model': 'elastic_net', 'phase': 'test'})
        self.assertGreater(diagnostics[0]['convergence_warning_count'], 0)
        self.assertEqual(diagnostics[0]['n_iter'], 1)

    def test_baseline_rank_and_stability_definitions(self):
        self.assertIsNone(metrics(np.arange(5), np.ones(5), correlation=False)['spearman'])
        coefficients = [np.array([2., 0.]), np.array([1., 0.]), np.array([1., 0.]),
                        np.array([2., 0.]), np.array([0., 0.])]
        available = [np.array([True, True])] * 5
        rows = feature_stability(coefficients, available, ['a', 'b'], ['A', 'B'])
        self.assertEqual(rows[0]['available_folds'], 5)
        self.assertEqual(rows[0]['nonzero_folds'], 4)
        self.assertEqual(rows[0]['stable_by_rule'], 1)

    def test_small_end_to_end_benchmark_has_complete_oof_predictions(self):
        rng = np.random.default_rng(4)
        X = rng.normal(size=(36, 8)).astype(np.float32)
        y = 50 + 7 * X[:, 0] + rng.normal(size=36)
        ids = np.array([f'd{i}' for i in range(36)])
        genes = np.array([f'g{i}' for i in range(8)])
        pools = np.repeat(np.arange(9), 4).astype(str)
        with tempfile.TemporaryDirectory() as temporary:
            output, report = Path(temporary) / 'local', Path(temporary) / 'report'
            summary = run_benchmark(X, y, ids, genes, genes, pools, output, report,
                                    max_features=6, outer_folds=3, inner_folds=2,
                                    max_iter=5000, make_plots=False)
            self.assertEqual(len(summary['metrics']), 5)
            self.assertTrue(summary['demo_model_roundtrip_verified'])
            with (output / 'oof_predictions.csv').open(newline='') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 36 * 5)
            for protocol, model in [('donor_cv', 'median'), ('donor_cv', 'ridge'),
                                    ('donor_cv', 'elastic_net'), ('pool_cv', 'median'), ('pool_cv', 'ridge')]:
                selected = [r for r in rows if r['protocol'] == protocol and r['model'] == model]
                self.assertEqual(len({r['donor_id'] for r in selected}), 36)
            manifest = json.loads((report / 'summary.json').read_text())
            self.assertEqual(manifest['n_donors'], 36)
            synthetic_data = Path(temporary) / 'synthetic_data'
            synthetic_data.mkdir()
            np.savez_compressed(synthetic_data / 'cd8_pseudobulk.npz', log1p_cpm=X, gene_ids=genes)
            with (synthetic_data / 'donor_metadata.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['donor_id', 'age', 'pool_number'])
                writer.writeheader()
                writer.writerows({'donor_id': d, 'age': float(a), 'pool_number': p} for d, a, p in zip(ids, y, pools))
            validate_training(synthetic_data, output, report)
            validate_training(synthetic_data, output, report)
            with self.assertRaises(FileExistsError):
                run_benchmark(X, y, ids, genes, genes, pools, output, report, make_plots=False)


if __name__ == '__main__':
    unittest.main()
