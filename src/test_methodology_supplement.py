"""Numerical audit tests and a tiny synthetic pool-EN supplement."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.stats import rankdata, spearmanr

from audit_cd8_features import basis, incremental_r2, partial_rank
from cell_pipeline import prepare_cell, read_json
from preprocess_cd8 import sha256
from run_cell_supplement import pool_elastic_net
from test_cell_pipeline import fixture, paths_for
from train_cd8 import load_data, run_benchmark


class FeatureAuditMathTests(unittest.TestCase):
    def test_unadjusted_rank_residual_equals_spearman(self):
        x = np.array([0., 0., 1., 3., 2., 8.])
        y = np.array([1., 1., 2., 5., 8., 9.])
        actual, rank = partial_rank(x, y, np.empty((len(x), 0)))
        self.assertAlmostEqual(actual, spearmanr(x, y).statistic)
        self.assertEqual(rank, 1)

    def test_rank_covariate_adjustment_reduces_simulated_shared_trend(self):
        rng = np.random.default_rng(42)
        z = rng.normal(size=400)
        x, y = z + rng.normal(scale=.15, size=400), z + rng.normal(scale=.15, size=400)
        marginal, _ = partial_rank(x, y, np.empty((400, 0)))
        # Callers rank continuous covariates, as in the real composition audit.
        covariate = rankdata(z)
        adjusted, _ = partial_rank(x, y, covariate[:, None])
        repeated, _ = partial_rank(x, y, np.column_stack([covariate, covariate]))
        self.assertGreater(marginal, .9)
        self.assertLess(abs(adjusted), .2)
        self.assertAlmostEqual(adjusted, repeated, places=10)

    def test_projection_rank_and_incremental_fit(self):
        z = np.linspace(-1, 1, 50)
        marker = (np.arange(50) % 2).astype(float)
        reduced = np.column_stack([np.ones(50), z])
        full = np.column_stack([reduced, marker])
        X = np.column_stack([2*z + 5*marker, z + np.sin(10*z)])
        raw, adjusted, parameters = incremental_r2(X, reduced, full)
        self.assertEqual(basis(np.column_stack([full, marker])).shape[1], 3)
        self.assertEqual(parameters, 1)
        self.assertAlmostEqual(raw[0], 1., places=9)
        self.assertAlmostEqual(adjusted[0], 1., places=9)


class SupplementalPoolTests(unittest.TestCase):
    def test_synthetic_fully_nested_pool_supplement_preserves_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, raw, shared, _ = fixture(root)
            paths = paths_for(root, 'B')
            prepare_cell('B', config, raw, paths)
            X, y, ids, genes, symbols, groups, source = load_data(paths['data'], paths['preprocessing'] / 'summary.json', paths['bundle'])
            ref_models, ref_report = root / 'reference_models', root / 'reference_report'
            run_benchmark(X, y, ids, genes, symbols, groups, ref_models, ref_report,
                          max_features=2, max_iter=5000, make_plots=False, source=source, cell_label='B')
            before = {p.name: sha256(p) for p in ref_models.iterdir() if p.is_file()}
            with patch('run_cell_supplement.ROOT', root), patch('run_cell_supplement.cell_paths', return_value=paths):
                result = pool_elastic_net('B', 'maximum', config, shared, ref_models, ref_report)
            self.assertEqual(result['n_donors'], 20)
            self.assertEqual(result['fit_count'], 140)
            self.assertEqual(len(result['selected_parameters']), 5)
            self.assertEqual([r['model'] for r in result['metrics']], ['median', 'ridge', 'elastic_net'])
            for name, digest in before.items(): self.assertEqual(sha256(ref_models/name), digest)
            saved = read_json(root / 'results/cells/B/pool_en_maximum_v1/validation.json')
            self.assertEqual(saved['status'], 'passed')
            self.assertEqual(saved['oof_rows'], 20)


if __name__ == '__main__':
    unittest.main()
