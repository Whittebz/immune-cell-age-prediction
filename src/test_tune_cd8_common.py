import contextlib
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.model_selection import KFold

from cell_pipeline import write_table
from train_cd8 import make_splits

from cell_pipeline import read_json
from tune_cd8_common import (DEFAULT_TUNING_CONFIG, MODEL_NAMES, age_bins, age_weights,
                            aggregate, candidate_list, fit_outer, load_checkpoint, macro_mae,
                            recorded_fit, validate_config, write_checkpoint)


def small_config():
    cfg=copy.deepcopy(read_json(DEFAULT_TUNING_CONFIG))
    cfg.update(feature_counts=[2,4],ridge_alphas=[.1,1.],elastic_net_alphas=[.1,1.],l1_ratios=[.1,.9],max_iter=5000)
    return cfg


def toy():
    rng=np.random.default_rng(42)
    y=np.linspace(19,97,40)
    X=rng.normal(size=(40,8));X[:,0]=y/20+rng.normal(scale=.1,size=40)
    ids=np.array([f'synthetic_{i:02d}' for i in range(40)])
    groups=np.repeat(np.arange(10).astype(str),4)
    genes=np.array([f'gene{i}' for i in range(8)])
    train,test=next(KFold(5,shuffle=True,random_state=42).split(X))
    inner=list(KFold(3,shuffle=True,random_state=43).split(X[train]))
    return X,y,ids,groups,genes,train,test,inner


class TuningPlanTests(unittest.TestCase):
    def test_locked_candidate_counts_and_tie_order(self):
        cfg=read_json(DEFAULT_TUNING_CONFIG);validate_config(cfg)
        counts=[len(candidate_list(name,cfg)) for name in MODEL_NAMES]
        self.assertEqual(counts,[18,45,45])
        self.assertEqual(2*5*(3*sum(counts)+3),3270)
        for name in MODEL_NAMES:
            values=candidate_list(name,cfg)
            key=lambda r:(r['max_features'],r['alpha'],r['l1_ratio'] or 0)
            self.assertEqual(values,sorted(values,key=key))
        self.assertEqual(candidate_list('en_expanded',cfg),candidate_list('en_age_weighted',cfg))

    def test_left_closed_bins_and_rejected_ages(self):
        cfg=small_config()
        np.testing.assert_array_equal(age_bins([18,39,40,59,60,79,80,100],cfg['age_bin_edges']),[0,0,1,1,2,2,3,3])
        for y in ([17],[101],[np.nan],[]):
            with self.assertRaises(ValueError):age_weights(y,cfg)

    def test_training_weights_partial_not_full_balancing(self):
        cfg=small_config();y=np.array([20,25,45,65,85.])
        w=age_weights(y,cfg)
        self.assertAlmostEqual(w.mean(),1)
        self.assertAlmostEqual(w[2]/w[0],np.sqrt(2))
        self.assertAlmostEqual(w[:2].sum()/w[2],np.sqrt(2))
        np.testing.assert_allclose(age_weights(np.array([50.,55.,59.]),cfg),1)
        self.assertAlmostEqual(macro_mae(y,np.full(5,40.),cfg),23.125)

    def test_both_en_preprocessing_and_train_only_weights(self):
        cfg=small_config();args=toy();X,y,ids,groups,genes,train,test,inner=args
        seen=[]
        def spy(model,A,target,diagnostics,context,weights=None,pipeline=False):
            if context['model']=='en_age_weighted':np.testing.assert_allclose(weights,age_weights(target,cfg))
            else:self.assertIsNone(weights)
            seen.append(context.copy())
            return recorded_fit(model,A,target,diagnostics,context,weights,pipeline)
        with contextlib.redirect_stdout(io.StringIO()),patch('tune_cd8_common.recorded_fit',side_effect=spy):
            arrays,detail=fit_outer(*args,cfg,'donor_cv',0)
        self.assertEqual(len(seen),63)
        self.assertEqual(len(detail['tuning']),60)
        self.assertEqual(len(detail['weight_audit']),4)
        self.assertEqual(len(detail['selected']),3)
        self.assertEqual(arrays['predictions'].shape,(3,len(test)))
        for mi in range(3):
            mask=arrays['available'][mi]
            reconstructed=((X[test][:,mask]-arrays['means'][mi,mask])/arrays['scales'][mi,mask])@arrays['coefficients'][mi,mask]+arrays['intercepts'][mi]
            np.testing.assert_allclose(reconstructed,arrays['predictions'][mi],atol=1e-10)
        # Same X partition/feature count yields exactly the same selector/scaler for both ENs.
        if detail['selected'][1]['max_features']==detail['selected'][2]['max_features']:
            np.testing.assert_array_equal(arrays['available'][1],arrays['available'][2])
            np.testing.assert_allclose(arrays['means'][1],arrays['means'][2])
            np.testing.assert_allclose(arrays['scales'][1],arrays['scales'][2])

    def test_outer_test_age_not_used_in_selection_or_prediction(self):
        cfg=small_config();args=list(toy())
        with contextlib.redirect_stdout(io.StringIO()):a,d=fit_outer(*args,cfg,'donor_cv',0)
        changed=args[1].copy();changed[args[6]]=98.;args[1]=changed
        with contextlib.redirect_stdout(io.StringIO()):b,e=fit_outer(*args,cfg,'donor_cv',0)
        self.assertEqual(d['selected'],e['selected'])
        self.assertEqual(d['tuning'],e['tuning'])
        self.assertEqual(d['weight_audit'],e['weight_audit'])
        for key in a:np.testing.assert_allclose(a[key],b[key])

    def test_equal_scores_choose_smallest_deterministic_candidate(self):
        cfg=small_config();args=list(toy());args[1]=np.full(40,50.)
        with contextlib.redirect_stdout(io.StringIO()):_,detail=fit_outer(*args,cfg,'donor_cv',0)
        for row in detail['selected']:
            self.assertEqual(row['max_features'],2);self.assertEqual(row['alpha'],.1)
            if row['model']!='ridge_expanded':self.assertEqual(row['l1_ratio'],.1)

    def test_complete_synthetic_aggregate_and_strict_json(self):
        cfg=small_config();X,y,ids,groups,genes,_,_,_=toy()
        plans={};checkpoints={};references=[];assignments=[]
        records=[{'donor_id':str(ids[i]),'age':float(y[i]),'sex':'female' if i%2 else 'male',
                  'pool_number':str(groups[i]),'n_cd8_cells':40+i,'actual_cd8_umi':100000+1000*i,'included':1} for i in range(len(y))]
        with contextlib.redirect_stdout(io.StringIO()):
            for protocol in cfg['protocols']:
                outer=make_splits(X,y,groups,5,42,protocol=='pool_cv');plans[protocol]={'outer':outer}
                for fold,(tr,te) in enumerate(outer):
                    inner=make_splits(X[tr],y[tr],groups[tr],3,43+fold,protocol=='pool_cv')
                    checkpoints[(protocol,fold)]=fit_outer(X,y,ids,groups,genes,tr,te,inner,cfg,protocol,fold)
                    for i in te:
                        assignments.append({'protocol':protocol,'donor_id':str(ids[i]),'fold':fold,'pool_number':str(groups[i])})
                        for name in ('median','ridge_v1','elastic_net_v1'):
                            # Synthetic reference only: train-median prediction, never test-age calibration.
                            value=float(np.median(y[tr]))
                            references.append({'protocol':protocol,'model':name,'donor_id':str(ids[i]),'fold':fold,
                                               'actual_age':float(y[i]),'predicted_age':value,'raw_age_gap':value-float(y[i])})
            with tempfile.TemporaryDirectory() as temp:
                private,report=Path(temp)/'private',Path(temp)/'report';private.mkdir();report.mkdir()
                write_table(private/'analysis_cohort.csv',records);write_table(private/'fold_assignments.csv',assignments)
                with patch('tune_cd8_common.reference_rows',return_value=references):
                    summary=aggregate(private,report,cfg,{'source':{},'protected_inputs':{}},ids,y,genes,genes,plans,checkpoints)
                self.assertEqual(summary['fit_count'],630)
                self.assertEqual(len(summary['metrics']),12)
                self.assertTrue(all(r['spearman'] is None for r in summary['metrics'] if r['model']=='median'))
                import json
                def bad_constant(value):raise AssertionError('Nonstandard JSON number: '+value)
                json.loads((report/'summary.json').read_text(encoding='utf-8'),parse_constant=bad_constant)
                self.assertTrue((report/'REPORT.md').is_file())
                self.assertTrue((report/'benchmark.svg').is_file())
                with np.load(private/'outer_models.npz',allow_pickle=False) as raw:
                    self.assertEqual(raw['coefficients'].shape,(6,5,8))

    def test_checkpoint_roundtrip_no_overwrite_and_tamper_rejection(self):
        args=toy();ids,train,test=args[2],args[5],args[6]
        arrays={'predictions':np.ones((3,len(test)))};detail={'note':'synthetic only'}
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp)/'fold_0'
            self.assertIsNone(load_checkpoint(directory,'donor_cv',0,ids,train,test))
            write_checkpoint(directory,'donor_cv',0,ids,train,test,arrays,detail)
            loaded,info=load_checkpoint(directory,'donor_cv',0,ids,train,test)
            np.testing.assert_array_equal(loaded['predictions'],arrays['predictions']);self.assertEqual(info,detail)
            with self.assertRaises(FileExistsError):write_checkpoint(directory,'donor_cv',0,ids,train,test,arrays,detail)
            (directory/'detail.json').write_text('{}',encoding='utf-8')
            with self.assertRaises(ValueError):load_checkpoint(directory,'donor_cv',0,ids,train,test)


if __name__=='__main__':unittest.main()
