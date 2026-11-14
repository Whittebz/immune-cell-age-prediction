"""Synthetic multicell integration tests. No four-lineage real-data training."""
import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import h5py
import numpy as np

from cell_pipeline import (DEFAULT_CONFIG, build_cohort, canonical_digest, check_prepared_protocol, package_cell,
                           prepare_cell, read_json, shared_protocol, train_cell, validate_cell)
from preprocess_cd8 import sha256
from train_cd8 import validate_fixed_splits


def synthetic_source(path, config):
    labels, donor, age, sex, pool = [], [], [], [], []
    all_subtypes = [label for group in config['cells'].values() for label in group['subtypes']]
    for d in range(20):
        for label in all_subtypes:
            # Donor0 lacks enough monocytes, while remaining eligible for B etc.
            if d == 0 and label == 'CD16 Mono':
                continue
            labels.append(label); donor.append(f'd{d:02d}'); age.append(20 + d * 3)
            sex.append('female' if d % 2 else 'male'); pool.append(str(d // 2))
    matrix = np.array([[1 + i % 7, 1 + (i // 3) % 5, 1 + (i // 5) % 3] for i in range(len(labels))], dtype=np.float32)
    string_type = h5py.string_dtype('utf-8')
    with h5py.File(path, 'w') as f:
        X = f.create_group('X'); X.attrs['encoding-type'] = 'csr_matrix'; X.attrs['shape'] = matrix.shape
        X.create_dataset('data', data=matrix.ravel())
        X.create_dataset('indices', data=np.tile(np.arange(3), len(matrix)))
        X.create_dataset('indptr', data=np.arange(0, matrix.size + 1, 3))
        obs = f.create_group('obs')
        for name, values in [('donor_id', donor), ('sex', sex), ('pool_number', pool), ('predicted.celltype.l2', labels)]:
            categories, codes = np.unique(np.asarray(values, dtype=str), return_inverse=True)
            group = obs.create_group(name)
            group.create_dataset('categories', data=categories.astype(object), dtype=string_type)
            group.create_dataset('codes', data=codes)
        for name, values in [('age', age), ('nFeature_RNA', np.full(len(labels), 3)),
                             ('nCount_RNA', matrix.sum(axis=1)), ('percent.mt', np.ones(len(labels))),
                             ('predicted.celltype.l2.score', np.full(len(labels), .9))]:
            obs.create_dataset(name, data=values)
        var = f.create_group('var'); var.attrs['_index'] = '_index'
        var.create_dataset('_index', data=['g1', 'g2', 'g3'], dtype=string_type)
        var.create_dataset('feature_name', data=['G1', 'G2', 'G3'], dtype=string_type)
    config['source_bytes'] = path.stat().st_size
    config['source_sha256'] = sha256(path)


def fixture(root):
    config = copy.deepcopy(read_json(DEFAULT_CONFIG))
    config['qc']['min_cells'] = 2; config['qc']['min_genes'] = 1
    config['training']['max_features'] = 2; config['training']['max_iter'] = 5000
    source = root / 'synthetic.h5ad'
    synthetic_source(source, config)
    shared = root / 'shared'
    summary = build_cohort(source, config, shared, root / 'common_summary.json', min_common=10)
    return config, source, shared, summary


def paths_for(root, cell):
    return {'data': root / 'data' / cell, 'preprocessing': root / 'prep' / cell,
            'models': root / 'models' / cell, 'report': root / 'report' / cell,
            'bundle': f'{cell.lower()}_pseudobulk.npz'}


class CellPipelineTests(unittest.TestCase):
    def test_config_and_frozen_cohort_are_traceable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, source, shared, summary = fixture(root)
            self.assertEqual(summary['n_common_donors'], 19)
            self.assertEqual(summary['cell_eligibility_metadata_only']['B']['eligible_donors'], 20)
            frozen, rows, plans = shared_protocol(config, shared)
            self.assertEqual(len(rows), 19)
            self.assertEqual(frozen['config_sha256'], canonical_digest(config))
            pools = np.array([r['pool_number'] for r in rows])
            for protocol in ('donor_cv', 'pool_cv'):
                outer = validate_fixed_splits(plans[protocol]['outer'], len(rows), pools if protocol == 'pool_cv' else None, 5)
                for fold, (train, _) in enumerate(outer):
                    validate_fixed_splits(plans[protocol]['inner'][fold], len(train), pools[train] if protocol == 'pool_cv' else None, 3)
            changed = copy.deepcopy(config); changed['qc']['min_cells'] = 1
            with self.assertRaises(ValueError): shared_protocol(changed, shared)
            with self.assertRaises(FileExistsError):
                build_cohort(source, config, shared, root / 'common_summary.json', min_common=10)

    def test_prepared_protocol_rejects_wrong_measurement_settings(self):
        config = read_json(DEFAULT_CONFIG)
        source = {'input': {'sha256': config['source_sha256']}, 'parameters': {
            'subtypes': config['cells']['B']['subtypes'], 'min_cells_per_donor': 30,
            'min_source_genes_per_cell': 200, 'max_source_mt_percent': 20,
            'cell_key': 'B', 'bundle_name': 'b_pseudobulk.npz',
            'subtype_column': config['annotation_column'], 'annotation_score_filter': None,
            'normalization': 'natural log1p(count / donor_actual_UMI * 1e6)'}}
        check_prepared_protocol('B', config, source)
        for key, value in [('cell_key', 'CD8'), ('annotation_score_filter', .9),
                           ('normalization', 'sum of already logged cells'), ('subtype_column', 'cell_type')]:
            wrong = copy.deepcopy(source); wrong['parameters'][key] = value
            with self.assertRaises(ValueError): check_prepared_protocol('B', config, wrong)

    def test_frozen_splits_reject_pool_or_row_leakage(self):
        with self.assertRaises(ValueError):
            validate_fixed_splits([([0, 1], [1, 2])], 3)
        with self.assertRaises(ValueError):
            validate_fixed_splits([([2, 3], [0, 1]), ([0, 1], [2, 3])], 4, np.array(['a','b','a','b']))

    def test_generic_cd4_preparation_preserves_six_subtypes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, source, _, _ = fixture(root)
            paths = paths_for(root, 'CD4')
            prepare_cell('CD4', config, source, paths)
            summary = read_json(paths['preprocessing'] / 'summary.json')
            self.assertEqual(summary['parameters']['cell_key'], 'CD4')
            self.assertEqual(summary['cells']['cd4_before_qc'], 120)
            self.assertEqual(summary['donors']['retained'], 20)
            self.assertTrue((paths['data'] / 'cd4_pseudobulk.npz').exists())
            with (paths['data'] / 'donor_metadata.csv').open(newline='') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(int(rows[0]['n_cd4_cells']), 6)
            self.assertEqual(len([k for k in rows[0] if k.startswith('fraction_')]), 6)

    def test_one_member_end_to_end_and_private_return_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, source, shared, _ = fixture(root)
            paths = paths_for(root, 'B')
            prepare_cell('B', config, source, paths)
            summary = train_cell('B', config, shared, paths)
            self.assertEqual(summary['n_donors'], 19)
            self.assertEqual(summary['run_metadata']['eligible_donors_before_common'], 20)
            self.assertEqual(summary['cell_label'], 'B')
            validate_cell('B', config, shared, paths, source)
            # Simulate receiving a leader-verified prepared bundle without raw H5AD.
            absent = root / 'not_distributed.h5ad'
            validate_cell('B', config, shared, paths, absent)
            with self.assertRaises(FileNotFoundError):
                package_cell('B', config, shared, paths, absent, root / 'returns')
            (paths['report'] / 'handoff_summary.md').write_text('SYNTHETIC_TEST_ONLY: completed protocol and limitations. ' * 8)
            (paths['report'] / 'biological_interpretation.md').write_text('SYNTHETIC_TEST_ONLY: not a real biological claim; needs human primary-literature review. ' * 8)
            archive_path = package_cell('B', config, shared, paths, absent, root / 'returns')
            with zipfile.ZipFile(archive_path) as archive:
                names = archive.namelist()
                self.assertTrue(any('oof_predictions.csv' in n for n in names))
                self.assertFalse(any(n.endswith('.h5ad') or 'pseudobulk.npz' in n or '.venv' in n for n in names))
                snapshot = json.loads(archive.read('B/cell_types.json'))
                self.assertEqual(snapshot['qc']['min_cells'], 2)
                self.assertIn('RETURN_MANIFEST.json', ' '.join(names))
            with self.assertRaises(FileExistsError):
                package_cell('B', config, shared, paths, absent, root / 'returns')


if __name__ == '__main__':
    unittest.main()
