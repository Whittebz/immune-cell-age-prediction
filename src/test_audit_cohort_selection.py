"""Small synthetic tests; run with -B to avoid bytecode outside the audit scope."""
import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from audit_cohort_selection import (
    OUTPUT, canonical_digest, failure_tables, prepare_output, retention_rows,
    standardized_difference, validate_metadata,
)


def fixture():
    rows = []
    for index, (age, sex, mono, cd8) in enumerate(((18, "female", 30, 30),
                                                (40, "male", 29, 30),
                                                (80, "female", 29, 29))):
        rows.append({"donor_id": f"synthetic_{index}", "age": str(age), "sex": sex,
                     "pool_number": "1", "n_cd4_cells": "30", "n_b_cells": "30",
                     "n_nk_cells": "30", "n_mono_cells": str(mono),
                     "n_cd8_cells": str(cd8), "common_eligible": str(int(mono >= 30 and cd8 >= 30))})
    return rows, [copy.deepcopy(rows[0])]


class CohortSelectionTests(unittest.TestCase):
    def test_canonical_hash_ignores_key_order_and_preserves_unicode(self):
        self.assertEqual(canonical_digest({"z": 1, "a": "年龄"}),
                         canonical_digest({"a": "年龄", "z": 1}))
        import hashlib
        self.assertEqual(canonical_digest({"a": "年龄", "z": 1}),
                         hashlib.sha256('{"a":"年龄","z":1}'.encode()).hexdigest())

    def test_threshold_is_inclusive_and_membership_checked(self):
        rows, common = fixture()
        age, sex, pool, counts, selected = validate_metadata(rows, common, 30)
        self.assertEqual(selected.tolist(), [True, False, False])
        common[0]["age"] = "19"
        with self.assertRaisesRegex(ValueError, "metadata disagree"):
            validate_metadata(rows, common, 30)

    def test_flags_must_agree_with_all_five_thresholds(self):
        rows, common = fixture()
        rows[1]["common_eligible"] = "1"
        with self.assertRaisesRegex(ValueError, "flags disagree"):
            validate_metadata(rows, common, 30)

    def test_exact_failure_patterns_and_pairwise_overlap(self):
        rows, common = fixture()
        _, _, _, counts, selected = validate_metadata(rows, common, 30)
        cells, patterns, overlaps = failure_tables(counts, selected, 30)
        self.assertEqual(sum(row["n_donors"] for row in patterns), 3)
        self.assertEqual(next(row["n_donors"] for row in patterns
                              if row["failed_cell_types"] == "MONO+CD8"), 1)
        self.assertEqual(next(row["n_fail_both"] for row in overlaps
                              if row["failed_type_a"] == "MONO" and row["failed_type_b"] == "CD8"), 1)
        self.assertEqual(next(row["n_excluded_solely_this_failure"] for row in cells
                              if row["cell_type"] == "MONO"), 1)

    def test_fixed_age_boundaries_and_denominators(self):
        age = np.array([18, 39, 40, 59, 60, 79, 80, 100])
        selected = np.array([True, False] * 4)
        rows = retention_rows(age, np.array(["female"] * 8), np.array(["1"] * 8), selected)
        bands = [row for row in rows if row["dimension"] == "age_band"]
        self.assertEqual([row["n_full"] for row in bands], [2] * 4)
        self.assertEqual([row["retention_percent"] for row in bands], [50.] * 4)

    def test_smd_sign_and_binary_formula(self):
        result = standardized_difference(np.array([3., 5.]), np.array([1., 3.]))
        self.assertAlmostEqual(result["smd"], np.sqrt(2))
        result = standardized_difference(np.array([1., 1., 0.]), np.array([1., 0., 0.]), binary=True)
        self.assertAlmostEqual(result["smd"], 1 / np.sqrt(2))
        self.assertEqual(standardized_difference(np.ones(3), np.ones(3))["smd"], 0.)

    def test_nonempty_output_refused_without_overwrite(self):
        # Any transient test data live only under the assigned audit directory.
        OUTPUT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="test_", dir=OUTPUT) as directory:
            destination = Path(directory) / "audit"
            prepare_output(destination)
            sentinel = destination / ".sentinel"
            sentinel.write_text("unchanged\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "nonempty"):
                prepare_output(destination)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "unchanged\n")


if __name__ == "__main__":
    unittest.main()
