import unittest
import numpy as np
from scipy.stats import rankdata, spearmanr
from interpret_cd4 import PARTS, align_common, association, composition_matrix, gene_index


class CD4InterpretationTest(unittest.TestCase):
    def metadata(self):
        result=[]
        for i in range(3):
            row={"donor_id":f"d{i}","age":str(30+i*20),"sex":"female","pool_number":"1","n_cd4_cells":"60","actual_cd4_umi":"600"}
            row.update({"fraction_"+p:str(1/6) for p in PARTS})
            result.append(row)
        return result

    def test_align_uses_common_order_and_rejects_wrong_labels(self):
        meta=self.metadata();cohort=[dict(meta[2]),dict(meta[0])]
        np.testing.assert_array_equal(align_common([r["donor_id"] for r in meta],meta,cohort),[2,0])
        cohort[0]["age"]="99"
        with self.assertRaises(ValueError):align_common([r["donor_id"] for r in meta],meta,cohort)
        with self.assertRaises(ValueError):align_common([r["donor_id"] for r in meta],meta,[meta[0],meta[0]])

    def test_composition_requires_complete_valid_fractions(self):
        meta=self.metadata();self.assertEqual(composition_matrix(meta).shape,(3,6))
        meta[0]["fraction_treg"]="0.9"
        with self.assertRaises(AssertionError):composition_matrix(meta)
        meta[0]["fraction_treg"]="nan"
        with self.assertRaises(ValueError):composition_matrix(meta)

    def test_gene_id_and_symbol_must_both_match(self):
        self.assertEqual(gene_index(["g1","g2"],["A","B"],"g2","B"),1)
        with self.assertRaises(ValueError):gene_index(["g1","g2"],["A","B"],"g2","A")

    def test_rank_adjustment_matches_independent_least_squares(self):
        rng=np.random.default_rng(72);comp=rng.dirichlet(np.ones(6),100)
        age=60*comp[:,0]+rng.normal(size=100);expression=2*age+10*comp[:,3]+rng.normal(size=100)
        marginal,adjusted,_=association(age,expression,comp)
        design=np.column_stack([np.ones(100)]+[rankdata(comp[:,j]) for j in [0,1,3,4,5]])
        def residual(v):
            v=rankdata(v);return v-design@np.linalg.lstsq(design,v,rcond=None)[0]
        expected=np.corrcoef(residual(age),residual(expression))[0,1]
        self.assertAlmostEqual(adjusted,expected,places=12)
        self.assertAlmostEqual(marginal,spearmanr(age,expression).statistic,places=12)
        with self.assertRaises(ValueError):association(age,np.ones(100),comp)


if __name__=="__main__":unittest.main()
