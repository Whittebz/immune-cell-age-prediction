#!/usr/bin/env python3
"""Post-hoc CD8 composition/feature diagnostics; frozen gene models stay intact."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re

import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from cell_pipeline import DEFAULT_CONFIG, read_json, read_table, shared_protocol, write_json, write_table
from preprocess_cd8 import ROOT, sha256
from train_cd8 import GRIDS, TopVarianceGenes, export_figure, load_data, make_splits, metrics, validate_fixed_splits

GENES = ('LRRN3', 'CDKN2A', 'CD70', 'IGFBP3')
PARTS = ('cd8_naive', 'cd8_tcm', 'cd8_tem', 'cd8_proliferating')
# Explicit marker panel, NOT a complete genomic X/Y chromosome annotation.
SEX_MARKERS = {'XIST', 'TSIX', 'JPX', 'RPS4Y1', 'RPS4Y2', 'DDX3Y', 'EIF1AY', 'UTY', 'KDM5D', 'ZFY', 'USP9Y', 'TMSB4Y', 'PRKY', 'PCDH11Y'}
RIBOSOMAL = re.compile(r'^(RPL(?:[0-9]+|P[0-9]+)|RPS[0-9]+)(?:A|B|X|Y[12]?)?$')


def basis(design):
    design = np.asarray(design, dtype=float)
    u, s, _ = np.linalg.svd(design, full_matrices=False)
    keep = s > (s.max() * max(design.shape) * np.finfo(float).eps if len(s) else 0)
    return u[:, keep]


def dummies(values):
    levels = np.unique(values)
    return np.column_stack([values == level for level in levels[1:]]).astype(float) if len(levels) > 1 else np.empty((len(values), 0))


def standardize(values):
    values = np.asarray(values, dtype=float)
    if values.ndim == 1: values = values[:, None]
    scale = values.std(axis=0)
    return (values - values.mean(axis=0)) / np.where(scale > 0, scale, 1)


def ranks(values):
    values = np.asarray(values, dtype=float)
    if values.ndim == 1: return rankdata(values)
    return np.column_stack([rankdata(values[:, j]) for j in range(values.shape[1])])


def partial_rank(x, y, covariates):
    covariates = np.asarray(covariates, dtype=float)
    if covariates.ndim == 1: covariates = covariates[:, None]
    design = np.column_stack([np.ones(len(x)), standardize(covariates)])
    q = basis(design)
    a, b = rankdata(x), rankdata(y)
    a, b = a - q @ (q.T @ a), b - q @ (q.T @ b)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    rho = float(a @ b / denominator) if denominator > 1e-12 else None
    return rho, q.shape[1]


def incremental_r2(X, reduced_design, full_design):
    """Incremental R² and df-adjusted improvement; no nuisance-as-cause claim."""
    X = np.asarray(X, dtype=float)
    q0, q1 = basis(reduced_design), basis(full_design)
    ss = np.sum(X * X, axis=0)
    sse0 = np.maximum(0, ss - np.sum((q0.T @ X) ** 2, axis=0))
    sse1 = np.maximum(0, ss - np.sum((q1.T @ X) ** 2, axis=0))
    eligible = sse0 > 1e-9
    result = np.full(X.shape[1], np.nan); adjusted = result.copy()
    result[eligible] = np.clip((sse0[eligible] - sse1[eligible]) / sse0[eligible], 0, 1)
    adjusted[eligible] = 1 - (sse1[eligible] / (len(X) - q1.shape[1])) / (sse0[eligible] / (len(X) - q0.shape[1]))
    return result, adjusted, q1.shape[1] - q0.shape[1]


def eta_squared(X, labels, variance):
    overall = X.mean(axis=0, dtype=np.float64)
    between = np.zeros(X.shape[1])
    for label in np.unique(labels):
        selected = labels == label
        between += selected.sum() * (X[selected].mean(axis=0, dtype=np.float64) - overall) ** 2
    return np.divide(between, len(X) * variance, out=np.zeros_like(between), where=variance > 1e-12)


def analyze(output):
    output = Path(output)
    if output.exists() and any(output.iterdir()): raise FileExistsError(output)
    cfg = read_json(DEFAULT_CONFIG)
    _, common, _ = shared_protocol(cfg, ROOT / 'data/shared')
    data, training, models = ROOT / 'data/processed/CD8', ROOT / 'results/cd8_training', ROOT / 'data/models/CD8'
    X, age, ids, gene_ids, symbols, pool, source = load_data(data, ROOT / 'results/cd8/summary.json')
    meta = read_table(data / 'donor_metadata.csv')
    sex = np.array([r['sex'] for r in meta]); cells = np.array([float(r['n_cd8_cells']) for r in meta])
    umi = np.array([float(r['actual_cd8_umi']) for r in meta])
    count_comp = np.column_stack([[float(r['fraction_' + part]) for r in meta] for part in PARTS])
    rna_comp = np.column_stack([[float(r['umi_' + part]) for r in meta] for part in PARTS]) / umi[:, None]
    np.testing.assert_allclose(count_comp.sum(axis=1), 1, atol=1e-10)
    np.testing.assert_allclose(rna_comp.sum(axis=1), 1, atol=1e-10)
    # Omit TEM as the reference component; all four proportions sum to one.
    chosen = [0, 1, 3]
    gene_indices = {}
    for symbol in GENES:
        indices = np.flatnonzero(symbols == symbol)
        if len(indices) != 1: raise ValueError('Ambiguous symbol requires an Ensembl-level selection')
        gene_indices[symbol] = int(indices[0])
    output.mkdir(parents=True, exist_ok=True)
    common_ids = {r['donor_id'] for r in common}
    associations = []
    for cohort_name, mask in (('cd8_maximum_977', np.ones(len(ids), dtype=bool)),
                              ('common_505_descriptive_only', np.array([name in common_ids for name in ids]))):
        n = int(mask.sum())
        technical = np.column_stack([dummies(sex[mask]), dummies(pool[mask]), ranks(np.log1p(cells[mask])), ranks(np.log1p(umi[mask]))])
        count_cov, rna_cov = ranks(count_comp[mask][:, chosen]), ranks(rna_comp[mask][:, chosen])
        specifications = {'marginal': np.empty((n, 0)), 'count_composition': count_cov,
                          'rna_composition': rna_cov, 'sex_pool_depth': technical,
                          'count_composition_plus_technical': np.column_stack([count_cov, technical]),
                          'rna_composition_plus_technical': np.column_stack([rna_cov, technical])}
        for symbol, index in gene_indices.items():
            for specification, covariates in specifications.items():
                rho, rank = partial_rank(X[mask, index], age[mask], covariates)
                associations.append({'cohort': cohort_name, 'gene_symbol': symbol, 'gene_id': gene_ids[index],
                                     'n': n, 'adjustment': specification, 'partial_rank_correlation': rho,
                                     'design_rank_with_intercept': rank})
    write_table(output / 'gene_age_adjusted_associations.csv', associations)
    # Reconstruct the frozen outer-training variance selectors, never fitting on their tests.
    fold_rows = read_table(models / 'fold_assignments.csv')
    availability = np.zeros((5, len(gene_ids)), dtype=bool)
    selected_rows, fold_summary = [], []
    first_plot = None
    marker_mask = np.array([s in SEX_MARKERS for s in symbols])
    ribosomal_mask = np.array([bool(RIBOSOMAL.fullmatch(s)) for s in symbols])
    mito_mask = np.char.startswith(symbols.astype(str), 'MT-')
    with threadpool_limits(limits=2):
        for fold in range(5):
            testing = {r['donor_id'] for r in fold_rows if r['protocol'] == 'donor_cv' and int(r['fold']) == fold}
            tr = np.array([i for i, name in enumerate(ids) if name not in testing])
            A = X[tr]
            select = TopVarianceGenes(1000).fit(A).indices_
            availability[fold, select] = True
            mean, variance, prevalence = A.mean(axis=0, dtype=np.float64), np.var(A, axis=0, dtype=np.float64), (A > 0).mean(axis=0)
            nonconstant = variance > 1e-12
            eta = eta_squared(A, pool[tr], variance)
            agez = standardize(age[tr]).ravel()
            depth = standardize(np.column_stack([np.log1p(cells[tr]), np.log1p(umi[tr])]))
            demographic = np.column_stack([np.ones(len(tr)), agez, agez ** 2, dummies(sex[tr]), depth])
            pool_partial, pool_adjusted, pool_df = incremental_r2(A, demographic, np.column_stack([demographic, dummies(pool[tr])]))
            without_sex = np.column_stack([np.ones(len(tr)), agez, agez ** 2, depth, dummies(pool[tr])])
            sex_partial, _, _ = incremental_r2(A, without_sex, np.column_stack([without_sex, dummies(sex[tr])]))
            for index in select:
                selected_rows.append({'outer_fold': fold, 'gene_id': gene_ids[index], 'gene_symbol': symbols[index],
                                      'train_n': len(tr), 'mean_log1p_cpm': mean[index], 'variance_log1p_cpm': variance[index],
                                      'donor_detection_fraction': prevalence[index], 'known_sex_marker_panel': int(marker_mask[index]),
                                      'ribosomal_symbol_pattern': int(ribosomal_mask[index]), 'mitochondrial_symbol_prefix': int(mito_mask[index]),
                                      'pool_eta_squared_unadjusted': eta[index], 'pool_partial_r2_given_age_sex_depth': pool_partial[index],
                                      'pool_df_adjusted_improvement': pool_adjusted[index],
                                      'sex_partial_r2_given_age_pool_depth': sex_partial[index]})
            fold_summary.append({'outer_fold': fold, 'train_n': len(tr), 'selected_genes': len(select),
                                 'nonconstant_genes': int(nonconstant.sum()),
                                 'all_nonconstant_mean_variance_spearman': float(spearmanr(mean[nonconstant], variance[nonconstant]).statistic),
                                 'selected_mean_variance_spearman': float(spearmanr(mean[select], variance[select]).statistic),
                                 'selected_detection_below_10pct': int((prevalence[select] < .1).sum()),
                                 'selected_known_sex_markers': int(marker_mask[select].sum()),
                                 'selected_ribosomal_pattern': int(ribosomal_mask[select].sum()),
                                 'selected_mitochondrial_prefix': int(mito_mask[select].sum()),
                                 'selected_median_pool_eta2': float(np.median(eta[select])),
                                 'selected_median_pool_partial_r2': float(np.nanmedian(pool_partial[select])),
                                 'selected_median_pool_df_adjusted_improvement': float(np.nanmedian(pool_adjusted[select])),
                                 'pool_added_parameters': pool_df,
                                 'selected_median_sex_partial_r2': float(np.nanmedian(sex_partial[select]))})
            if first_plot is None: first_plot = (mean, variance, select, nonconstant)
            print(f'Feature diagnostics fold {fold+1}/5: {fold_summary[-1]}', flush=True)
    saved_stability = {r['gene_id']: int(r['available_folds']) for r in read_table(training / 'elastic_net_feature_stability.csv')}
    for index in np.flatnonzero(availability.any(axis=0)):
        if int(availability[:, index].sum()) != saved_stability[gene_ids[index]]:
            raise ValueError('Reconstructed selectors differ from frozen feature availability')
    write_table(output / 'selected_training_genes.csv', selected_rows)
    write_table(output / 'feature_diagnostics_by_fold.csv', fold_summary)
    # Small, separately named composition-only predictive comparators on identical OOF folds.
    supplemental_private = ROOT / 'data/models/CD8/composition_diagnostics_v1'
    if supplemental_private.exists() and any(supplemental_private.iterdir()): raise FileExistsError(supplemental_private)
    supplemental_private.mkdir(parents=True, exist_ok=True)
    composition_scores, comp_predictions = [], []
    with threadpool_limits(limits=2):
        for protocol, grouped in (('donor_cv', False), ('pool_cv', True)):
            folds = {r['donor_id']: int(r['fold']) for r in fold_rows if r['protocol'] == protocol}
            outer = [(np.array([i for i, name in enumerate(ids) if folds[name] != k]), np.array([i for i, name in enumerate(ids) if folds[name] == k])) for k in range(5)]
            validate_fixed_splits(outer, len(ids), pool if grouped else None, 5)
            for name, composition in (('composition_count_ridge', count_comp[:, chosen]), ('composition_rna_ridge', rna_comp[:, chosen])):
                prediction = np.full(len(ids), np.nan)
                for fold, (tr, te) in enumerate(outer):
                    inner = make_splits(composition[tr], age[tr], pool[tr], 3, 42 + fold + 1, grouped)
                    pipe = Pipeline([('scale', StandardScaler()), ('regressor', Ridge(solver='cholesky'))])
                    search = GridSearchCV(pipe, {'regressor__alpha': GRIDS['ridge']['alpha']}, scoring='neg_mean_absolute_error', cv=inner, n_jobs=1)
                    search.fit(composition[tr], age[tr]); prediction[te] = search.predict(composition[te])
                    for i in te: comp_predictions.append({'protocol': protocol, 'model': name, 'donor_id': ids[i], 'fold': fold,
                                                         'actual_age': float(age[i]), 'predicted_age': float(prediction[i]), 'raw_age_gap': float(prediction[i]-age[i])})
                composition_scores.append({'protocol': protocol, 'model': name, **metrics(age, prediction)})
    write_table(supplemental_private / 'oof_predictions.csv', comp_predictions)
    write_table(output / 'composition_only_metrics.csv', composition_scores)
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    mean, variance, select, nonconstant = first_plot
    background = np.flatnonzero(nonconstant)[::3]
    axes[0].scatter(mean[background], variance[background], s=4, alpha=.12, color='#888888', label='Other nonconstant genes (thinned)')
    axes[0].scatter(mean[select], variance[select], s=7, alpha=.5, color='#bf7340', label='Top-1000 in outer train fold 0')
    axes[0].set(xlabel='Mean log1pCPM', ylabel='Variance log1pCPM', title='Mean–variance selection diagnostic'); axes[0].legend(fontsize=8)
    for column, label in (('selected_detection_below_10pct', '<10% donors detected'), ('selected_known_sex_markers', 'Known sex-marker panel'), ('selected_ribosomal_pattern', 'Ribosomal name pattern')):
        axes[1].plot(range(1, 6), [r[column] for r in fold_summary], marker='o', label=label)
    axes[1].set(xlabel='Outer training fold', ylabel='Count among 1000 selected genes', title='Descriptive flags, not automatic exclusions'); axes[1].legend(fontsize=8)
    export_figure(fig, output / 'feature_selection_diagnostics.png'); plt.close(fig)
    result = {'status': 'passed', 'n_cd8_donors': len(ids), 'genes_examined': list(GENES), 'adjusted_associations': associations,
              'selected_union_genes': int(availability.any(axis=0).sum()), 'frozen_variance_selectors_reproduced': True,
              'fold_diagnostics': fold_summary, 'composition_only_metrics': composition_scores,
              'source_sha256': {'bundle': sha256(data / 'cd8_pseudobulk.npz'), 'metadata': sha256(data / 'donor_metadata.csv'),
                                'fold_assignments': sha256(models / 'fold_assignments.csv'),
                                'feature_stability': sha256(training / 'elastic_net_feature_stability.csv')},
              'limitations': ['Adjustment is linear regression of ranked gene/age on ranked continuous covariates plus sex/pool indicators; no p-values or causal independence claim.',
                              'TEM omitted as compositional reference; count fractions and actual-UMI fractions analyzed separately.',
                              'Age/sex/depth-adjusted pool association may reflect residual biology; it does not prove technical noise.',
                              'Sex marker panel is not a complete X/Y chromosome annotation. Ribosomal symbols are a heuristic, not a full housekeeping annotation.',
                              'No low-prevalence/HVG filter added to frozen main training, and no alternative selector performance claimed.',
                              'Common505 gene associations are descriptive only, not a common505 retrained gene model.',
                              'Composition-only Ridge is an information comparator, not causal mediation or matched model-capacity decomposition.',
                              'Available annotation is CD8 TEM, not a separate validated TEMRA category.']}
    write_json(output / 'summary.json', result)
    lines = ['# CD8 亚群组成与训练折特征审查', '', '不修改原基因模型；所有新增分析独立保存。', '',
             '## 基因—年龄的条件关联', '对年龄和基因表达取秩，回归去除协变量后计算残差 Pearson 相关；连续组成/深度协变量取秩，性别/pool 使用虚拟变量。',
             '四类组成之和为1，省略 TEM 作为参照。分别使用细胞数量比例与实际 UMI 比例。这里只检验所列条件下的描述性关联，不证明细胞内在机制。', '',
             '| 基因 | 未调整ρ（977人） | 数量组成调整 | RNA组成调整 | RNA组成+性别/pool/深度 |', '|---|---:|---:|---:|---:|']
    for gene in GENES:
        values = {r['adjustment']: r['partial_rank_correlation'] for r in associations if r['cohort'] == 'cd8_maximum_977' and r['gene_symbol'] == gene}
        lines.append(f'| {gene} | {values["marginal"]:.3f} | {values["count_composition"]:.3f} | {values["rna_composition"]:.3f} | {values["rna_composition_plus_technical"]:.3f} |')
    lines += ['', '## 仅组成的预测对照', '同原977人、相同外层划分，内层三折选择 Ridge alpha；只用三个亚群比例，不用基因表达、年龄或技术变量作为输入。',
              '| 验证 | 输入 | MAE（年） | R² |', '|---|---|---:|---:|']
    for r in composition_scores: lines.append(f'| {r["protocol"]} | {r["model"]} | {r["mae_years"]:.3f} | {r["r2"]:.3f} |')
    lines += ['', '## 特征选择核查', f'精确复现原5个外层训练折的筛选可用频次，合计 {result["selected_union_genes"]} 个基因。',
              '逐折表包含检出率、均值—方差相关、已知性别标记、核糖体名称规则和 pool 关联。已知性别标记不等于全部 X/Y 基因；pool 关联不自动等于批次噪声。',
              'pool 条件 R²按年龄及年龄平方、性别、log细胞数和log实际UMI调整，同时提供增加自由度后的描述性改善，避免将多参数拟合本身当作强证据。',
              '不据这些事后结果改主特征列表，也不声称已经比较过新的HVG/低表达过滤器。', '', '## 解释限制',
              '若组成调整后关联衰减，支持组成参与解释；若仍残留关联，也不能证明独立的细胞内衰老机制。',
              '不能把 CD8 TEM 注释直接改称 TEMRA，也不能把两个预测模型的R²比值解释为组成的因果中介比例。']
    (output / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'gene_associations': [r for r in associations if r['cohort']=='cd8_maximum_977'], 'feature_summary': fold_summary,
                      'composition_only_metrics': composition_scores}, ensure_ascii=False, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/methodology_audit/cd8_features')
    args = parser.parse_args(); analyze(args.output)
