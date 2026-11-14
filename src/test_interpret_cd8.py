"""Tests for descriptive summaries, not evidence of biological mechanisms."""
import json
import unittest

import numpy as np

from interpret_cd8 import CANDIDATES, ROOT, age_trends, bibtex_authors, describe_gene, rank_correlation


class InterpretationTests(unittest.TestCase):
    def test_conditional_direction_is_distinct_from_marginal_correlation(self):
        evidence = {
            'gene_id': 'g1', 'gene_symbol': 'G1',
            'mean_signed_standardized_coefficient': '-0.5',
            'available_folds': '5', 'nonzero_folds': '5',
            'sign_consistency_when_nonzero': '1.0',
        }
        summary = describe_gene(np.array([20, 40, 60]), np.array([1., 2., 3.]),
                                np.array([1, 2, 3]), evidence)
        self.assertEqual(summary['model_direction'], 'lower_prediction')
        self.assertEqual(summary['same_cohort_age_spearman_descriptive'], 1.0)
        self.assertEqual(summary['n_donors_detected'], 3)
        self.assertEqual(summary['total_gene_umi'], 6)

    def test_age_bands_partition_boundaries(self):
        result = age_trends(np.array([19, 39, 40, 59, 60, 79, 80, 97]), np.arange(8.), 'G')
        self.assertEqual([r['n_donors'] for r in result], [2, 2, 2, 2])
        self.assertEqual(result[0]['median_log1p_cpm'], 0.5)
        self.assertEqual(sum(r['n_donors'] for r in result), 8)

    def test_missing_or_constant_data_is_not_an_invented_correlation(self):
        self.assertIsNone(rank_correlation(np.ones(4), np.arange(4)))
        with self.assertRaises(ValueError):
            describe_gene(np.arange(3), np.arange(2), np.arange(3), {})

    def test_bibliographic_author_format(self):
        self.assertEqual(bibtex_authors('Peters MJ, Joehanes R, Pilling LC, et al.'),
                         'Peters, M. J. and Joehanes, R. and Pilling, L. C. and others')
        self.assertEqual(bibtex_authors('Serrano M, Hannon GJ, Beach D.'),
                         'Serrano, M. and Hannon, G. J. and Beach, D.')
        self.assertEqual(bibtex_authors('Tesselaar K, Arens R, van Schijndel GM, et al.'),
                         'Tesselaar, K. and Arens, R. and van Schijndel, G. M. and others')

    def test_reference_registry_has_traceable_ids_and_boundaries(self):
        registry = json.loads((ROOT / 'docs/CD8_BIOLOGY_REFERENCES.json').read_text(encoding='utf-8'))
        references = {r['id']: r for r in registry['references']}
        self.assertEqual(len(references), len(registry['references']))
        for symbol, _, reference_ids, _ in CANDIDATES:
            for reference_id in reference_ids:
                record = references[reference_id]
                self.assertIn(symbol, record['genes'])
                self.assertEqual(record['url'], 'https://doi.org/' + record['doi'])
                self.assertTrue(record['pmid'].isdigit())
                self.assertTrue(record['supports_zh'])
                self.assertTrue(record['does_not_support_zh'])
        self.assertIn('合计转录本', references['R2']['does_not_support_zh'])
        self.assertIn('血清蛋白', references['R6']['does_not_support_zh'])


if __name__ == '__main__':
    unittest.main()
