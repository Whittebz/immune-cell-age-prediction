#!/usr/bin/env python3
"""Add maximum-cohort or nested pool-Elastic-Net analyses without changing main_v1.

Never alter frozen engines, shared manifests, original model artifacts or scores.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from cell_pipeline import (DEFAULT_CONFIG, cell_paths, check_prepared_protocol, read_json,
                           read_table, shared_protocol, write_json, write_table)
from preprocess_cd8 import ROOT, sha256
from train_cd8 import (GRIDS, build_regressor, feature_stability, fit_recorded, load_data,
                       make_preprocessor, make_splits, metrics, run_benchmark, tune_models,
                       validate_fixed_splits)
from validate_cd8_training import validate as validate_training


def inputs(cell, cohort_mode, config, shared):
    paths = cell_paths(cell)
    X, y, ids, genes, symbols, groups, source = load_data(paths['data'], paths['preprocessing'] / 'summary.json', paths['bundle'])
    check_prepared_protocol(cell, config, source)
    validation = read_json(paths['preprocessing'] / 'validation.json')
    if validation['status'] != 'passed' or not validation['independent_source_spot_checks']:
        raise ValueError('Source-verified preprocessing is required')
    metadata = {r['donor_id']: r for r in read_table(paths['data'] / 'donor_metadata.csv')}
    plans, frozen = None, None
    if cohort_mode == 'common':
        frozen, cohort, plans = shared_protocol(config, shared)
        index_of = {name: i for i, name in enumerate(ids)}
        index = np.array([index_of[r['donor_id']] for r in cohort], dtype=int)
        X, y, ids, groups = X[index], y[index], ids[index], groups[index]
        if not np.array_equal(y, [float(r['age']) for r in cohort]) or not np.array_equal(groups, [r['pool_number'] for r in cohort]):
            raise ValueError('Shared cohort and prepared metadata differ')
    records = [metadata[name] for name in ids]
    return paths, X, y, ids, genes, symbols, groups, source, records, plans, frozen


def maximum(cell, config, shared):
    p, X, y, ids, genes, symbols, groups, source, records, _, _ = inputs(cell, 'maximum', config, shared)
    private = ROOT / 'data/models' / cell / 'maximum_v1'
    report = ROOT / 'results/cells' / cell / 'maximum_v1'
    t = config['training']
    run_benchmark(X, y, ids, genes, symbols, groups, private, report,
                  seed=t['seed'], max_features=t['max_features'], outer_folds=t['outer_folds'],
                  inner_folds=t['inner_folds'], max_iter=t['max_iter'], cell_label=cell,
                  source=source, cohort_records=records,
                  run_metadata={'cell_key': cell, 'cohort_mode': 'maximum_v1',
                                'note': 'All lineage-eligible donors; not a fair raw score ranking across different lineages.'})
    validate_training(p['data'], private, report, p['bundle'])


def pool_elastic_net(cell, mode, config, shared, reference_models=None, reference_report=None):
    p, X, y, ids, genes, symbols, groups, source, records, plans, frozen = inputs(cell, mode, config, shared)
    if mode == 'common':
        ref_private, ref_report = p['models'], p['report']
    elif cell == 'CD8':
        ref_private, ref_report = ROOT / 'data/models/CD8', ROOT / 'results/cd8_training'
    else:
        ref_private = ROOT / 'data/models' / cell / 'maximum_v1'
        ref_report = ROOT / 'results/cells' / cell / 'maximum_v1'
    ref_private = Path(reference_models) if reference_models else ref_private
    ref_report = Path(reference_report) if reference_report else ref_report
    validate_training(p['data'], ref_private, ref_report, p['bundle'])
    reference_summary = read_json(ref_report / 'summary.json')
    if reference_summary['n_donors'] != len(ids):
        raise ValueError('Reference and supplemental cohorts differ')
    assignments = [r for r in read_table(ref_private / 'fold_assignments.csv') if r['protocol'] == 'pool_cv']
    fold_of = {r['donor_id']: int(r['fold']) for r in assignments}
    if len(assignments) != len(ids) or set(fold_of) != set(ids):
        raise ValueError('Reference pool fold coverage mismatch')
    t = config['training']
    outer = [(np.array([i for i, name in enumerate(ids) if fold_of[name] != k]),
              np.array([i for i, name in enumerate(ids) if fold_of[name] == k])) for k in range(t['outer_folds'])]
    outer = validate_fixed_splits(outer, len(ids), groups, t['outer_folds'])
    if plans is not None:
        approved = validate_fixed_splits(plans['pool_cv']['outer'], len(ids), groups, t['outer_folds'])
        if any(not np.array_equal(a, b) for pair, expected in zip(outer, approved) for a, b in zip(pair, expected)):
            raise ValueError('Reference outer folds differ from shared protocol')
    name = f'pool_en_{mode}_v1'
    private = ROOT / 'data/models' / cell / name
    report = ROOT / 'results/cells' / cell / name
    for directory in (private, report):
        if directory.exists() and any(directory.iterdir()): raise FileExistsError(directory)
    private.mkdir(parents=True, exist_ok=True); report.mkdir(parents=True, exist_ok=True)
    diagnostics, tuning, new_oof, fold_rows, coefficients, availability, selected_parameters = [], [], [], [], [], [], []
    prediction = np.full(len(y), np.nan)
    with threadpool_limits(limits=2):
        for fold, (train, test) in enumerate(outer):
            print(f'[{cell} pool EN {mode}] fold {fold+1}/5 train={len(train)} test={len(test)}', flush=True)
            inner = (make_splits(X[train], y[train], groups[train], t['inner_folds'], t['seed'] + fold + 1, True)
                     if plans is None else validate_fixed_splits(plans['pool_cv']['inner'][fold], len(train), groups[train], t['inner_folds']))
            params = tune_models(X[train], y[train], inner, ('elastic_net',), t['max_features'], diagnostics,
                                 {'protocol': 'pool_cv', 'fold': fold}, max_iter=t['max_iter'], seed=t['seed'], tuning_rows=tuning)['elastic_net']
            pipe = Pipeline(make_preprocessor(t['max_features']).steps + [('regressor', build_regressor('elastic_net', params, t['max_iter'], t['seed']))])
            fit_recorded(pipe, X[train], y[train], diagnostics,
                         {'protocol': 'pool_cv', 'fold': fold, 'phase': 'outer_refit', 'model': 'elastic_net', 'params': params})
            prediction[test] = pipe.predict(X[test])
            selected = pipe.named_steps['genes'].indices_
            coef = np.zeros(len(genes)); coef[selected] = pipe.named_steps['regressor'].coef_
            present = np.zeros(len(genes), dtype=bool); present[selected] = True
            coefficients.append(coef); availability.append(present)
            selected_parameters.append({'fold': fold, **params})
            fold_rows.append({'protocol': 'pool_cv', 'model': 'elastic_net', 'fold': fold, **metrics(y[test], prediction[test])})
            for i in test:
                new_oof.append({'protocol': 'pool_cv', 'model': 'elastic_net', 'fold': fold, 'donor_id': str(ids[i]),
                                'actual_age': float(y[i]), 'predicted_age': float(prediction[i]), 'raw_age_gap': float(prediction[i] - y[i])})
            print(f'  MAE={fold_rows[-1]["mae_years"]:.4f}; selected={params}', flush=True)
    if not np.isfinite(prediction).all(): raise ValueError('Incomplete/nonfinite OOF predictions')
    # A fresh consistency path, and comparisons on the identical pool folds.
    baseline_rows = [r for r in read_table(ref_private / 'oof_predictions.csv') if r['protocol'] == 'pool_cv']
    if any(r['model'] not in ('median', 'ridge') for r in baseline_rows):
        raise ValueError('Supplement expects reference median/Ridge pool rows only')
    lookup = {name: i for i, name in enumerate(ids)}
    for row in baseline_rows:
        i = lookup[row['donor_id']]
        if float(row['actual_age']) != y[i] or int(row['fold']) != fold_of[ids[i]]:
            raise ValueError('Reference OOF age/fold differs')
    for fold, (train, test) in enumerate(outer):
        for row in [r for r in baseline_rows if r['model'] == 'median' and int(r['fold']) == fold]:
            np.testing.assert_allclose(float(row['predicted_age']), np.median(y[train]), atol=1e-12)
    scores = []
    for model in ('median', 'ridge'):
        ref_rows = [r for r in baseline_rows if r['model'] == model]
        if len(ref_rows) != len(ids) or {r['donor_id'] for r in ref_rows} != set(ids): raise ValueError('Reference OOF incomplete')
        target = np.array([float(r['actual_age']) for r in ref_rows]); pred = np.array([float(r['predicted_age']) for r in ref_rows])
        scores.append({'protocol': 'pool_cv', 'model': model, **metrics(target, pred, model != 'median')})
        for fold in range(t['outer_folds']):
            part = [r for r in ref_rows if int(r['fold']) == fold]
            fold_rows.append({'protocol': 'pool_cv', 'model': model, 'fold': fold,
                              **metrics(np.array([float(r['actual_age']) for r in part]), np.array([float(r['predicted_age']) for r in part]), model != 'median')})
    scores.append({'protocol': 'pool_cv', 'model': 'elastic_net', **metrics(y, prediction)})
    fold_summary = []
    for model in ('median', 'ridge', 'elastic_net'):
        rows = [r for r in fold_rows if r['model'] == model]
        for key in ('mae_years', 'rmse_years', 'r2'):
            vals = np.array([r[key] for r in rows])
            fold_summary.append({'protocol': 'pool_cv', 'model': model, 'metric': key, 'fold_mean': float(vals.mean()), 'fold_sd_ddof1': float(vals.std(ddof=1)), 'folds': len(vals)})
    stability = feature_stability(coefficients, availability, genes, symbols)
    write_table(private / 'oof_predictions.csv', new_oof)
    write_table(private / 'fold_assignments.csv', assignments)
    write_table(private / 'inner_tuning_scores.csv', tuning)
    write_json(private / 'fit_diagnostics.json', diagnostics)
    np.savez_compressed(private / 'outer_coefficients.npz', coefficients=np.stack(coefficients), available=np.stack(availability), gene_ids=genes)
    write_table(report / 'metrics.csv', scores); write_table(report / 'fold_metrics.csv', fold_rows)
    write_table(report / 'fold_mean_sd.csv', fold_summary); write_table(report / 'elastic_net_feature_stability.csv', stability)
    warnings = sum(r['convergence_warning_count'] for r in diagnostics)
    outer_warnings = sum(r['convergence_warning_count'] for r in diagnostics if r['phase'] == 'outer_refit')
    summary = {'status': 'completed_with_convergence_warnings' if warnings else 'completed', 'cell': cell,
               'cohort': mode, 'n_donors': len(ids), 'n_pools': len(np.unique(groups)), 'metrics': scores,
               'method': 'Fully nested grouped 5x3 CV; original fixed grids; selector/scaler fit only inner training; same reference outer pool folds.',
               'grids': GRIDS, 'selected_parameters': selected_parameters, 'fit_count': len(diagnostics),
               'convergence_warnings_all_fits': warnings, 'convergence_warnings_outer_refits': outer_warnings,
               'stable_features_by_rule': sum(r['stable_by_rule'] for r in stability),
               'source_sha256': {'bundle': sha256(p['data'] / p['bundle']), 'reference_oof': sha256(ref_private / 'oof_predictions.csv'),
                                 'reference_assignments': sha256(ref_private / 'fold_assignments.csv')},
               'private_artifacts': {path.name: {'sha256': sha256(path), 'bytes': path.stat().st_size} for path in private.iterdir()},
               'notes': ['Pooled metrics are primary; fold mean and sample SD are secondary, not CI.',
                         'No full-data hyperparameters reused for held-out scoring; no score-driven extra tuning.',
                         'No old artifacts overwritten; no independent external-cohort or clinical-age validation.']}
    write_json(report / 'summary.json', summary)
    validation = {'status': 'passed', 'oof_rows': len(new_oof), 'unique_donors': len({r['donor_id'] for r in new_oof}),
                  'same_reference_outer_pool_folds': True, 'held_out_pool_disjointness': True, 'train_median_baseline_verified': True,
                  'pooled_metrics_recomputed': True, 'scope': 'Saved OOF/split consistency; not independent external-cohort validation.'}
    write_json(report / 'validation.json', validation)
    lines = [f'# {cell} Elastic Net pool 留出补充验证', '', f'队列：{mode}；供者数：{len(ids)}。',
             '外层按 pool 五折，内层按 pool 三折；使用原参数网格，不从全队列拟合结果借用参数。', '',
             '| 模型 | pooled OOF MAE（年） | RMSE（年） | R² |', '|---|---:|---:|---:|']
    for row in scores: lines.append(f'| {row["model"]} | {row["mae_years"]:.3f} | {row["rmse_years"]:.3f} | {row["r2"]:.3f} |')
    lines += ['', f'总拟合 {len(diagnostics)} 次；候选/外层合计收敛警告 {warnings}，外层重拟合警告 {outer_warnings}。',
              '完整警告保存在私有 fit_diagnostics.json；不静默剔除候选。',
              '逐折 Mean ± sample SD 另表提供；不能将其当作置信区间或独立重复实验。',
              '这些是当前队列的批次留出结果，不是跨研究队列或临床验证。']
    (report / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('maximum', 'pool-en'))
    parser.add_argument('--cell', choices=('CD4', 'CD8', 'B', 'NK', 'MONO'), required=True)
    parser.add_argument('--cohort', choices=('common', 'maximum'), default='common')
    parser.add_argument('--shared', type=Path, default=ROOT / 'data/shared')
    parser.add_argument('--reference-models', type=Path)
    parser.add_argument('--reference-report', type=Path)
    args = parser.parse_args(); cfg = read_json(DEFAULT_CONFIG)
    if args.action == 'maximum': maximum(args.cell, cfg, args.shared)
    else: pool_elastic_net(args.cell, args.cohort, cfg, args.shared, args.reference_models, args.reference_report)
