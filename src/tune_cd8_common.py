#!/usr/bin/env python3
"""Independent, exploratory CD8 tuning; never replace the frozen common_v1 run."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import platform
from importlib.metadata import version
from pathlib import Path
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from audit_oof_diagnostics import compute_tables, validate_and_join
from cell_pipeline import (DEFAULT_CONFIG, canonical_digest, read_json, read_table,
                           shared_protocol, write_json, write_table)
from preprocess_cd8 import ROOT, sha256
from run_cell_supplement import inputs
from train_cd8 import (build_regressor, export_figure, feature_stability,
                       make_preprocessor, validate_fixed_splits)

DEFAULT_TUNING_CONFIG = ROOT / 'config/cd8_tuning_v2.json'
MODEL_NAMES = ('ridge_expanded', 'en_expanded', 'en_age_weighted')
AGE_LABELS = ('18-39', '40-59', '60-79', '80-100')


def environment_versions():
    return {'python': platform.python_version(), **{name: version(name) for name in ('numpy', 'scipy', 'scikit-learn', 'matplotlib', 'threadpoolctl')}}


def clean_numeric(value):
    if isinstance(value, dict): return {key: clean_numeric(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)): return [clean_numeric(item) for item in value]
    if isinstance(value, np.generic): value = value.item()
    return None if isinstance(value, float) and not np.isfinite(value) else value


def ids_digest(ids):
    return canonical_digest([str(x) for x in ids])


def age_bins(y, edges):
    y = np.asarray(y, dtype=float)
    edges = np.asarray(edges, dtype=float)
    if y.ndim != 1 or not len(y) or not np.isfinite(y).all():
        raise ValueError('Expected finite nonempty ages')
    if len(edges) != 5 or np.any(np.diff(edges) <= 0) or np.any(y < edges[0]) or np.any(y >= edges[-1]):
        raise ValueError('Ages outside the four fixed left-closed bins')
    return np.searchsorted(edges[1:-1], y, side='right')


def age_weights(y, cfg):
    bins = age_bins(y, cfg['age_bin_edges'])
    counts = np.bincount(bins, minlength=4)
    w = counts[bins].astype(float) ** (-float(cfg['age_weight_power']))
    w /= w.mean()
    if not np.isfinite(w).all() or np.any(w <= 0): raise ValueError('Invalid training weights')
    return w


def macro_mae(y, pred, cfg):
    bins = age_bins(y, cfg['age_bin_edges'])
    error = np.abs(np.asarray(pred) - y)
    return float(np.mean([error[bins == b].mean() for b in range(4) if np.any(bins == b)]))


def validate_config(cfg):
    if cfg['models'] != list(MODEL_NAMES) or cfg['protocols'] != ['donor_cv', 'pool_cv']:
        raise ValueError('Unexpected model/protocol plan')
    if cfg['outer_folds'] != 5 or cfg['inner_folds'] != 3 or cfg['tol'] != 1e-4:
        raise ValueError('Unexpected folds/tolerance')
    for key in ('feature_counts', 'ridge_alphas', 'elastic_net_alphas', 'l1_ratios'):
        values = cfg[key]
        if not values or values != sorted(set(values)) or min(values) <= 0: raise ValueError(key)
    if any(int(k) != k for k in cfg['feature_counts']) or max(cfg['l1_ratios']) > 1:
        raise ValueError('Invalid feature counts/l1 ratios')
    if cfg['age_bin_edges'] != [18, 40, 60, 80, 101] or cfg['age_weight_power'] != .5:
        raise ValueError('Age weighting must match the locked study')
    if cfg['max_iter'] < 1 or cfg['full_data_refit'] is not False: raise ValueError('Invalid fitting budget')
    expected = 'unweighted mean of three inner-fold global MAEs for every strategy'
    if cfg['inner_selection'] != expected: raise ValueError('Do not change the scoring objective mid-study')


def candidate_list(model, cfg):
    if model not in MODEL_NAMES: raise ValueError(model)
    if model == 'ridge_expanded':
        return [dict(max_features=int(k), alpha=float(a), l1_ratio=None)
                for k in cfg['feature_counts'] for a in cfg['ridge_alphas']]
    return [dict(max_features=int(k), alpha=float(a), l1_ratio=float(r))
            for k in cfg['feature_counts'] for a in cfg['elastic_net_alphas'] for r in cfg['l1_ratios']]


def estimator(model, params, cfg):
    ridge = model == 'ridge_expanded'
    kwargs = {'alpha': params['alpha']}
    if not ridge: kwargs['l1_ratio'] = params['l1_ratio']
    return build_regressor('ridge' if ridge else 'elastic_net', kwargs, cfg['max_iter'], cfg['seed'])


def recorded_fit(model, X, y, diagnostics, context, weights=None, pipeline=False):
    kwargs = {} if weights is None else {('regressor__sample_weight' if pipeline else 'sample_weight'): weights}
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter('always')
        model.fit(X, y, **kwargs)
    reg = model.named_steps['regressor'] if pipeline else model
    diagnostics.append({**context, 'train_n': len(y),
                        'n_iter': int(np.max(reg.n_iter_)) if getattr(reg, 'n_iter_', None) is not None else None,
                        'dual_gap': float(np.max(reg.dual_gap_)) if getattr(reg, 'dual_gap_', None) is not None else None,
                        'convergence_warning_count': sum(issubclass(w.category, ConvergenceWarning) for w in seen),
                        'warnings': [{'category': w.category.__name__, 'message': str(w.message)} for w in seen]})
    return model


def weight_record(y, train_ids, valid_ids, cfg, context):
    bins = age_bins(y, cfg['age_bin_edges']); counts = np.bincount(bins, minlength=4)
    w = age_weights(y, cfg)
    return {**context, 'train_n': len(y), 'valid_n': len(valid_ids),
            'train_ids_sha256': ids_digest(train_ids), 'valid_ids_sha256': ids_digest(valid_ids),
            **{f'age_n_{label.replace("-", "_")}': int(counts[i]) for i, label in enumerate(AGE_LABELS)},
            'weight_min': float(w.min()), 'weight_max': float(w.max()), 'weight_mean': float(w.mean()),
            'effective_n': float(w.sum() ** 2 / (w @ w)),
            'weights_sha256': hashlib.sha256(w.astype('<f8').tobytes()).hexdigest()}


def fit_outer(X, y, ids, groups, genes, train, test, inner, cfg, protocol, fold):
    """Only train and INNER validation ages influence selection. Test is prediction-only."""
    Xtrain, ytrain, train_ids = X[train], y[train], ids[train]
    candidates = {model: candidate_list(model, cfg) for model in MODEL_NAMES}
    scores = {model: [[] for _ in candidates[model]] for model in MODEL_NAMES}
    tuning, diagnostics, weight_audit, selector_audit = [], [], [], []
    for inner_fold, (itr, iva) in enumerate(inner):
        common = {'protocol': protocol, 'outer_fold': fold, 'inner_fold': inner_fold}
        w = age_weights(ytrain[itr], cfg)
        weight_audit.append(weight_record(ytrain[itr], train_ids[itr], train_ids[iva], cfg, common))
        for k in cfg['feature_counts']:
            prep = make_preprocessor(k)
            Ztrain = prep.fit_transform(Xtrain[itr], ytrain[itr])
            Zvalid = prep.transform(Xtrain[iva])
            indices = prep.named_steps['genes'].indices_
            selector_audit.append({**common, 'stage': 'inner', 'max_features': k,
                                   'selected_genes': len(indices), 'train_n': len(itr), 'valid_n': len(iva),
                                   'train_ids_sha256': ids_digest(train_ids[itr]), 'valid_ids_sha256': ids_digest(train_ids[iva]),
                                   'selected_gene_ids_sha256': ids_digest(genes[indices])})
            for name in MODEL_NAMES:
                for ci, params in enumerate(candidates[name]):
                    if params['max_features'] != k: continue
                    context = {**common, 'phase': 'inner_tuning', 'model': name, **params}
                    reg = recorded_fit(estimator(name, params, cfg), Ztrain, ytrain[itr], diagnostics, context,
                                       w if name == 'en_age_weighted' else None)
                    pred = reg.predict(Zvalid)
                    if not np.isfinite(pred).all(): raise ValueError('Nonfinite candidate predictions; do not discard silently')
                    error = float(np.mean(np.abs(pred - ytrain[iva])))
                    scores[name][ci].append(error)
                    tuning.append({**common, 'model': name, **params, 'mae_years': error,
                                   'macro_age_mae_years': macro_mae(ytrain[iva], pred, cfg),
                                   'train_n': len(itr), 'valid_n': len(iva)})
        print(f'[{protocol} fold {fold+1}] inner {inner_fold+1}/3 complete', flush=True)
    selected, prediction, coefficients, available, means, scales, intercepts = [], [], [], [], [], [], []
    weight_audit.append(weight_record(ytrain, train_ids, ids[test], cfg,
                                      {'protocol': protocol, 'outer_fold': fold, 'inner_fold': -1}))
    for name in MODEL_NAMES:
        if any(len(values) != 3 for values in scores[name]): raise ValueError('Incomplete inner candidate coverage')
        best = int(np.argmin([np.mean(values) for values in scores[name]]))
        params = candidates[name][best]
        selected.append({'protocol': protocol, 'model': name, 'outer_fold': fold, **params,
                         'mean_inner_mae_years': float(np.mean(scores[name][best]))})
        pipe = Pipeline(make_preprocessor(params['max_features']).steps + [('regressor', estimator(name, params, cfg))])
        context = {'protocol': protocol, 'outer_fold': fold, 'inner_fold': -1,
                   'phase': 'outer_refit', 'model': name, **params}
        recorded_fit(pipe, Xtrain, ytrain, diagnostics, context,
                     age_weights(ytrain, cfg) if name == 'en_age_weighted' else None, pipeline=True)
        pred = pipe.predict(X[test])
        if not np.isfinite(pred).all(): raise ValueError('Nonfinite outer predictions')
        prediction.append(pred)
        indices = pipe.named_steps['genes'].indices_
        coef = np.zeros(X.shape[1]); coef[indices] = pipe.named_steps['regressor'].coef_
        mask = np.zeros(X.shape[1], dtype=bool); mask[indices] = True
        mean = np.zeros(X.shape[1]); mean[indices] = pipe.named_steps['scale'].mean_
        scale = np.ones(X.shape[1]); scale[indices] = pipe.named_steps['scale'].scale_
        coefficients.append(coef); available.append(mask); means.append(mean); scales.append(scale)
        intercepts.append(float(pipe.named_steps['regressor'].intercept_))
        selector_audit.append({'protocol': protocol, 'outer_fold': fold, 'inner_fold': -1,
                               'stage': 'outer', 'model': name, 'max_features': params['max_features'],
                               'selected_genes': len(indices), 'train_n': len(train), 'valid_n': len(test),
                               'train_ids_sha256': ids_digest(train_ids), 'valid_ids_sha256': ids_digest(ids[test]),
                               'selected_gene_ids_sha256': ids_digest(genes[indices])})
        # No outer errors are computed here; only the selected algorithm predicts the held-out rows.
        print(f'[{protocol} fold {fold+1}] selected {name}: {params}; inner MAE={selected[-1]["mean_inner_mae_years"]:.4f}', flush=True)
    arrays = {'predictions': np.stack(prediction), 'coefficients': np.stack(coefficients),
              'available': np.stack(available), 'means': np.stack(means), 'scales': np.stack(scales),
              'intercepts': np.asarray(intercepts)}
    detail = {'tuning': tuning, 'diagnostics': diagnostics, 'weight_audit': weight_audit,
              'selector_audit': selector_audit, 'selected': selected}
    return arrays, detail


def protected_paths(config_path):
    paths = [config_path, Path(__file__), DEFAULT_CONFIG, ROOT/'src/run_cell_supplement.py', ROOT/'src/audit_oof_diagnostics.py', ROOT/'requirements.txt']
    paths += [ROOT/'src'/name for name in ('preprocess_cd8.py', 'train_cd8.py', 'cell_pipeline.py', 'validate_cd8.py', 'validate_cd8_training.py')]
    paths += [ROOT/'data/shared'/name for name in ('common_cohort.csv', 'eligibility.csv', 'splits.json', 'protocol.json')]
    paths += [ROOT/'data/processed/CD8'/name for name in ('cd8_pseudobulk.npz', 'donor_metadata.csv', 'gene_metadata.csv')]
    paths += [ROOT/'results/cd8'/name for name in ('summary.json', 'validation.json')]
    for directory in (ROOT/'data/models/CD8', ROOT/'data/models/CD8/common_v1', ROOT/'data/models/CD8/pool_en_common_v1',
                      ROOT/'data/models/CD8/pool_en_maximum_v1', ROOT/'results/cd8_training',
                      ROOT/'results/cells/CD8/common_v1', ROOT/'results/cells/CD8/pool_en_common_v1',
                      ROOT/'results/cells/CD8/pool_en_maximum_v1', ROOT/'results/cells/CD8/audit_common_v1'):
        paths.extend(p for p in directory.iterdir() if p.is_file())
    return sorted(set(p.resolve() for p in paths))


def verify_protected(manifest):
    for name, digest in manifest['protected_inputs'].items():
        if sha256(Path(name)) != digest: raise ValueError(f'Protected baseline/input changed: {name}')


def load_checkpoint(directory, protocol, fold, ids, train, test):
    marker = directory/'complete.json'
    if not marker.exists(): return None
    status = read_json(marker)
    if status['protocol'] != protocol or status['outer_fold'] != fold or status['train_ids_sha256'] != ids_digest(ids[train]) or status['test_ids_sha256'] != ids_digest(ids[test]):
        raise ValueError('Checkpoint split mismatch')
    for name, digest in status['files'].items():
        if sha256(directory/name) != digest: raise ValueError('Checkpoint hash mismatch')
    with np.load(directory/'arrays.npz', allow_pickle=False) as raw:
        arrays = {name: raw[name] for name in raw.files}
    return arrays, read_json(directory/'detail.json')


def write_checkpoint(directory, protocol, fold, ids, train, test, arrays, detail):
    if directory.exists() and any(directory.iterdir()): raise FileExistsError(f'Unsealed partial checkpoint: {directory}')
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory/'arrays.npz', **arrays)
    write_json(directory/'detail.json', detail)
    write_json(directory/'complete.json', {'protocol': protocol, 'outer_fold': fold,
               'train_ids_sha256': ids_digest(ids[train]), 'test_ids_sha256': ids_digest(ids[test]),
               'files': {name: sha256(directory/name) for name in ('arrays.npz', 'detail.json')}})


def reference_rows():
    rows = read_table(ROOT/'data/models/CD8/common_v1/oof_predictions.csv') + read_table(ROOT/'data/models/CD8/pool_en_common_v1/oof_predictions.csv')
    names = {'median': 'median', 'ridge': 'ridge_v1', 'elastic_net': 'elastic_net_v1'}
    return [{**r, 'model': names[r['model']]} for r in rows]


def aggregate(private, report, cfg, manifest, ids, y, genes, symbols, plans, checkpoints):
    rows = reference_rows(); tuning, diagnostics, weights, selectors, selected = [], [], [], [], []
    labels = [f'{protocol}/{name}' for protocol in cfg['protocols'] for name in MODEL_NAMES]
    shape = (len(labels), 5, len(genes))
    coef = np.zeros(shape); available = np.zeros(shape, dtype=bool); means = np.zeros(shape); scales = np.ones(shape)
    intercepts = np.zeros((len(labels), 5))
    for (protocol, fold), (arrays, detail) in checkpoints.items():
        test = np.asarray(plans[protocol]['outer'][fold][1], dtype=int)
        for mi, name in enumerate(MODEL_NAMES):
            for i, pred in zip(test, arrays['predictions'][mi]):
                rows.append({'protocol': protocol, 'model': name, 'donor_id': str(ids[i]), 'fold': fold,
                             'actual_age': float(y[i]), 'predicted_age': float(pred), 'raw_age_gap': float(pred-y[i])})
            gi = labels.index(f'{protocol}/{name}')
            for target, key in ((coef, 'coefficients'), (available, 'available'), (means, 'means'), (scales, 'scales')):
                target[gi, fold] = arrays[key][mi]
            intercepts[gi, fold] = arrays['intercepts'][mi]
        for target, key in ((tuning, 'tuning'), (diagnostics, 'diagnostics'), (weights, 'weight_audit'),
                            (selectors, 'selector_audit'), (selected, 'selected')):
            target.extend(detail[key])
    records = read_table(private/'analysis_cohort.csv'); assignments = read_table(private/'fold_assignments.csv')
    groups, _ = validate_and_join([rows], records, assignments, cell='CD8')
    tables, _ = compute_tables(groups, cell='CD8')
    for row in tables['fold_metrics_recomputed']:
        protocol, model, fold = row['protocol'], row['model'], row['fold']
        current = [r for r in groups[(protocol, model)] if r['fold'] == fold]
        baseline_fold = next(r for r in tables['fold_metrics_recomputed'] if r['protocol']==protocol and r['model']=='median' and r['fold']==fold)
        row['mae_delta_vs_median_years'] = row['mae_years'] - baseline_fold['mae_years']
        row['macro_age_mae_years'] = macro_mae(np.array([r['age'] for r in current]), np.array([r['pred'] for r in current]), cfg)
    tables = clean_numeric(tables)
    scores = tables['pooled_metrics']
    for score in scores:
        parts = [r for r in tables['age_context_metrics'] if r['protocol']==score['protocol'] and r['model']==score['model'] and r['n']]
        score['macro_age_mae_years'] = float(np.mean([r['mae_years'] for r in parts]))
        current = groups[(score['protocol'],score['model'])]
        truth, prediction = np.array([r['age'] for r in current]), np.array([r['pred'] for r in current])
        score['oof_pred_vs_actual_slope'] = float(np.dot(truth-truth.mean(), prediction-prediction.mean()) / np.sum((truth-truth.mean())**2))
        score['oof_mean_signed_residual'] = float(np.mean(prediction-truth))
    for score in scores:
        reference = next(r for r in scores if r['protocol']==score['protocol'] and r['model']=='elastic_net_v1')
        score['mae_delta_vs_v1_en_years'] = score['mae_years']-reference['mae_years']
    write_table(private/'oof_predictions.csv', rows)
    write_table(private/'inner_tuning_scores.csv', tuning)
    write_json(private/'fit_diagnostics.json', diagnostics)
    write_table(private/'age_weight_audit.csv', weights)
    # Outer-only fields are blank in inner audit rows; use a uniform table schema.
    for row in selectors: row.setdefault('model', '')
    selector_keys = list(selectors[0])
    write_table(private/'selector_audit.csv', [{k:r.get(k,'') for k in selector_keys} for r in selectors])
    np.savez_compressed(private/'outer_models.npz', coefficients=coef, available=available, means=means,
                        scales=scales, intercepts=intercepts, gene_ids=genes, group_labels=np.asarray(labels))
    write_table(report/'metrics.csv', scores)
    write_table(report/'fold_metrics.csv', tables['fold_metrics_recomputed'])
    write_table(report/'fold_summary.csv', tables['fold_summary'])
    write_table(report/'age_band_metrics.csv', tables['age_context_metrics'])
    write_table(report/'sex_stratified_metrics.csv', tables['sex_stratified_metrics'])
    write_table(report/'error_depth_associations.csv', tables['error_depth_associations'])
    write_table(report/'selected_parameters.csv', selected)
    stable_counts = {}
    for gi, label in enumerate(labels):
        stable = feature_stability(coef[gi], available[gi], genes, symbols)
        write_table(report/(label.replace('/', '__')+'_feature_stability.csv'), stable)
        stable_counts[label] = sum(r['stable_by_rule'] for r in stable)
    convergence = []
    for protocol in cfg['protocols']:
        for name in MODEL_NAMES:
            current = [r for r in diagnostics if r['protocol']==protocol and r['model']==name]
            convergence.append({'protocol': protocol, 'model': name, 'fits': len(current),
                                'inner_warning_messages': sum(r['convergence_warning_count'] for r in current if r['phase']=='inner_tuning'),
                                'outer_warning_messages': sum(r['convergence_warning_count'] for r in current if r['phase']=='outer_refit')})
    write_table(report/'convergence_summary.csv', convergence)
    verify_protected(manifest)
    summary = {'status': 'completed_with_convergence_warnings' if any(r['convergence_warning_count'] for r in diagnostics) else 'completed',
               'experiment_id': cfg['experiment_id'], 'n_donors': len(ids), 'n_input_genes': len(genes),
               'n_pools': len({r['pool_number'] for r in records}), 'metrics': scores, 'configuration': cfg,
               'fit_count': len(diagnostics), 'convergence_warning_count_all_fits': sum(r['convergence_warning_count'] for r in diagnostics),
               'convergence_warning_count_outer_refits': sum(r['convergence_warning_count'] for r in diagnostics if r['phase']=='outer_refit'),
               'convergence_by_strategy': convergence, 'source': manifest['source'], 'selected_parameters': selected,
               'stable_features_by_group': stable_counts,
               'private_artifacts': {p.name:{'sha256':sha256(p),'bytes':p.stat().st_size} for p in private.iterdir() if p.is_file()},
               'limitations': ['Exploratory after inspecting v1 outer OOF; not an untouched confirmatory or external test.',
                               'Only CD8 tuned: v1 remains the uniform five-cell main comparison.',
                               'Square-root frequency weighting is partial age balancing, not equal age-bin weight.',
                               'All strategies retain global inner MAE selection; macro-age MAE is descriptive secondary.',
                               'v1-v2 changes search space and iteration budget together; no single-factor causal claim.',
                               'No full-data winner refit, no true-age residual correction, no outer-score-based strategy selection.']}
    plots(report, scores, tables['age_context_metrics'], groups, cfg)
    lines = ['# CD8共同505人独立调优（探索性敏感性）', '',
             '保留原v1基准，不把调优结果混入只用原方案训练的其他细胞排名。冻结505人及全部内外层分折不变。',
             '每个训练折内独立筛选/标准化；两种EN使用同一网格和全体供者MAE内层评分。年龄权重仅进入训练回归损失，按当次训练年龄计算；新样本预测不输入年龄。',
             '平方根反频率仅部分均衡，组总权重正比样本数平方根，不是四组完全等权。未根据外层分数扩展网格或拼接“各折最优策略”。', '',
             '| 验证 | 策略 | pooled MAE年 | R² | 宏平均年龄组MAE年 | 相对v1 EN的MAE差 |', '|---|---|---:|---:|---:|---:|']
    for r in scores:
        lines.append(f'| {r["protocol"]} | {r["model"]} | {r["mae_years"]:.3f} | {r["r2"]:.3f} | {r["macro_age_mae_years"]:.3f} | {r["mae_delta_vs_v1_en_years"]:+.3f} |')
    lines += ['', '负的MAE差表示当前划分下比v1 EN误差更小；不是显著性证明。逐年龄组同时报告n、误差和高估/低估，不能只看总体均值。',
              f'新回归模型记录拟合{len(diagnostics)}次；收敛警告{summary["convergence_warning_count_all_fits"]}，其中外层{summary["convergence_warning_count_outer_refits"]}。原始候选和警告未静默丢弃。',
              '先前OOF已被观察，本研究仍是探索性调优，而非新独立验证集。五折训练集重叠，Mean±sample SD不是CI或独立生物学重复。',
              '未生成全数据赢家模型，也未利用全505人的真实年龄校正预测误差。', '',
              '![总体误差](<benchmark.png>)', '![分年龄组误差](<age_band_comparison.png>)',
              '![供者CV预测](<donor_predictions.png>)', '',
              '所有逐供者OOF、模型数值和分折审计只在被Git忽略的data目录，不公开行级资料。']
    (report/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    verify_protected(manifest)
    # Completion marker is last, so resume never mistakes an unfinished report for a finished run.
    write_json(report/'summary.json', summary)
    return summary


def plots(report, scores, age_rows, groups, cfg):
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'.cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    order = ('median','ridge_v1','elastic_net_v1','ridge_expanded','en_expanded','en_age_weighted')
    colors = ('#bbbbbb','#888888','#476c93','#9a8a68','#c58440','#528c74')
    fig, axes = plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    for axis, protocol in zip(axes,cfg['protocols']):
        values=[next(r['mae_years'] for r in scores if r['protocol']==protocol and r['model']==name) for name in order]
        axis.bar(range(len(order)),values,color=colors);axis.set_xticks(range(len(order)),order,rotation=35,ha='right',fontsize=8)
        axis.set(ylabel='Pooled OOF MAE (years)',title=protocol+' | same 505 donors')
    export_figure(fig,report/'benchmark.png');plt.close(fig)
    fig, axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    for axis,protocol in zip(axes,cfg['protocols']):
        for name in ('elastic_net_v1','ridge_expanded','en_expanded','en_age_weighted'):
            values=[next(r['mae_years'] for r in age_rows if r['protocol']==protocol and r['model']==name and r['age_band']==band) for band in AGE_LABELS]
            axis.plot(range(4),values,marker='o',label=name)
        axis.set_xticks(range(4),AGE_LABELS);axis.set(ylabel='OOF MAE (years)',title=protocol+' | age groups');axis.legend(fontsize=8)
    export_figure(fig,report/'age_band_comparison.png');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4.3),constrained_layout=True)
    all_pred = [r['pred'] for name in ('elastic_net_v1','en_expanded','en_age_weighted') for r in groups[('donor_cv',name)]]
    limits = (float(np.floor((min(18,min(all_pred))-5)/5)*5), float(np.ceil((max(100,max(all_pred))+5)/5)*5))
    for axis,name in zip(axes,('elastic_net_v1','en_expanded','en_age_weighted')):
        rows=groups[('donor_cv',name)];y=np.array([r['age'] for r in rows]);p=np.array([r['pred'] for r in rows])
        axis.scatter(y,p,s=10,alpha=.35,color='#476c93');axis.plot([18,100],[18,100],'--',color='black',linewidth=1)
        axis.set(xlabel='Actual age',ylabel='OOF predicted age',title=name,xlim=(18,100),ylim=limits)
    export_figure(fig,report/'donor_predictions.png');plt.close(fig)


def run(config_path=DEFAULT_TUNING_CONFIG, run_id='tuning_v2', resume=False):
    config_path=Path(config_path).resolve();cfg=read_json(config_path);validate_config(cfg)
    if not run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in run_id): raise ValueError('Unsafe run id')
    private=ROOT/'data/models/CD8'/run_id;report=ROOT/'results/cells/CD8'/run_id
    base_cfg=read_json(DEFAULT_CONFIG);frozen,cohort,plans=shared_protocol(base_cfg,ROOT/'data/shared')
    p,X,y,ids,genes,symbols,groups,source,records,_,_=inputs('CD8','common',base_cfg,ROOT/'data/shared')
    baseline = read_json(ROOT/'results/cells/CD8/common_v1/summary.json')
    if baseline['n_donors'] != 505 or read_json(ROOT/'results/cells/CD8/common_v1/validation.json')['status'] != 'passed':
        raise ValueError('A verified common505 baseline is required')
    for name, item in baseline['private_artifacts'].items():
        if sha256(ROOT/'data/models/CD8/common_v1'/name) != item['sha256']:
            raise ValueError('Reference baseline artifact changed')
    if len(ids)!=505 or not np.array_equal(ids,[r['donor_id'] for r in cohort]): raise ValueError('Expected exact canonical505')
    manifest_path=private/'run_manifest.json'
    if resume:
        manifest=read_json(manifest_path)
        if manifest['config_sha256']!=canonical_digest(cfg) or manifest['code_sha256']!=sha256(Path(__file__)) or manifest['run_id']!=run_id or manifest['versions']!=environment_versions():
            raise ValueError('Resume must use identical code/config/run-id/environment')
        verify_protected(manifest)
        if (report/'summary.json').exists():
            summary=read_json(report/'summary.json')
            for name,item in summary['private_artifacts'].items():
                if sha256(private/name)!=item['sha256']: raise ValueError('Completed run artifact changed')
            print('Run already complete; preserved without refitting.');return summary
    else:
        for directory in (private,report):
            if directory.exists() and any(directory.iterdir()): raise FileExistsError(directory)
        private.mkdir(parents=True,exist_ok=True);report.mkdir(parents=True,exist_ok=True)
        manifest={'experiment_id':cfg['experiment_id'],'run_id':run_id,'created_at_utc':datetime.now(timezone.utc).isoformat(),
                  'config':cfg,'config_sha256':canonical_digest(cfg),'code_sha256':sha256(Path(__file__)), 'versions':environment_versions(),
                  'protected_inputs':{str(path):sha256(path) for path in protected_paths(config_path)},
                  'n_donors':len(ids),'n_input_genes':len(genes),
                  'source':{'cohort_sha256':frozen['cohort_sha256'],'splits_sha256':frozen['splits_sha256'],
                            'config_canonical_sha256':frozen['config_sha256'],'bundle_sha256':sha256(p['data']/p['bundle'])},
                  'interpretation':cfg['interpretation']}
        write_json(manifest_path,manifest)
        # Byte-preserving copies; source files have already been read/validated.
        for name in ('analysis_cohort.csv','fold_assignments.csv'):
            (private/name).write_bytes((ROOT/'data/models/CD8/common_v1'/name).read_bytes())
        write_json(report/'TUNING_PLAN.json',manifest)
    checkpoints={}
    with threadpool_limits(limits=2):
        for protocol in cfg['protocols']:
            outer=validate_fixed_splits(plans[protocol]['outer'],len(y),groups if protocol=='pool_cv' else None,5)
            for fold,(train,test) in enumerate(outer):
                directory=private/'checkpoints'/protocol/f'fold_{fold}'
                cached=load_checkpoint(directory,protocol,fold,ids,train,test) if resume else None
                if cached is not None:
                    print(f'[{protocol} fold {fold+1}] verified checkpoint reused',flush=True);checkpoints[(protocol,fold)]=cached;continue
                inner=validate_fixed_splits(plans[protocol]['inner'][fold],len(train),groups[train] if protocol=='pool_cv' else None,3)
                print(f'[{protocol} fold {fold+1}/5] train={len(train)} test={len(test)}',flush=True)
                result=fit_outer(X,y,ids,groups,genes,train,test,inner,cfg,protocol,fold)
                write_checkpoint(directory,protocol,fold,ids,train,test,*result)
                checkpoints[(protocol,fold)]=result
        summary=aggregate(private,report,cfg,manifest,ids,y,genes,symbols,plans,checkpoints)
    print(json.dumps({'status':summary['status'],'fit_count':summary['fit_count'],'metrics':summary['metrics'],
                      'convergence_warning_count_all_fits':summary['convergence_warning_count_all_fits'],
                      'convergence_warning_count_outer_refits':summary['convergence_warning_count_outer_refits']},indent=2),flush=True)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=DEFAULT_TUNING_CONFIG)
    parser.add_argument('--run-id',default='tuning_v2')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();run(args.config,args.run_id,args.resume)
