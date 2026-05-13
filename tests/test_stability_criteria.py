"""
Tests for Stability_Criteria (analysis_utils.py)
=================================================

Background
----------
Stability_Criteria encodes *where* on the electrochemical Pourbaix diagram
a material must be stable.  It is initialised with:
  - Us  : a single potential (V vs RHE) or a [low, high] interval
  - pHs : a single pH value or a [low, high] interval
  - decomposition_threshold : max allowed ΔG_decomp (eV/atom, default 0.5)

`max_dG_in_region(decom_G)` receives a 2-D numpy array of precomputed decomposition
energies, shaped (31, 19) to match the fixed grids:
  - U_interval  = np.linspace(-2, 4,  31)  →  31 potential points, step 0.2 V
  - pH_interval = np.linspace(-2, 16, 19)  →  19 pH points,       step 1.0

It returns the maximum decomposition energy in the selected U×pH region (float).
A material is stable when max_dG_in_region(decom_G) <= decomposition_threshold.

Index reference used throughout these tests
-------------------------------------------
  U  =  0.0  V  →  row index 10   (= (0 - (-2)) / 0.2)
  U  =  1.0  V  →  row index 15
  pH =  7.0      →  col index  9   (= (7 - (-2)) / 1.0)
  pH =  6.0      →  col index  8
  pH =  8.0      →  col index 10

When a *range* [lo, hi] is given, Stability_Criteria finds the floor index
of lo and the ceil index of hi (both inclusive), then slices with Python's
exclusive upper bound, so Us=[0, 1] → slice [10:16] covers U = 0.0…1.0.

Test order
----------
1. Constructor validation  — bad inputs raise ValueError before any evaluation
2. Scalar point evaluation — simplest case: one U, one pH
3. Threshold behaviour     — boundary value and custom threshold
4. U-range evaluation      — list Us, scalar pH
5. pH-range evaluation     — scalar U, list pHs
6. 2-D range evaluation    — list Us AND list pHs
"""

import numpy as np
import pytest
from gnome_aqueous_stability.analysis_utils import Stability_Criteria

SHAPE = (31, 19)  # shape of the decom_G array: (U points, pH points)


# ── Helper factories ─────────────────────────────────────────────────────────

def _all_above(threshold=0.5):
    """Return a decom_G array where every cell is ABOVE threshold (unstable)."""
    return np.full(SHAPE, threshold + 0.3)


def _all_below(threshold=0.5):
    """Return a decom_G array where every cell is BELOW threshold (stable)."""
    return np.full(SHAPE, threshold - 0.2)


# ── 1. Constructor validation ─────────────────────────────────────────────────

def test_invalid_Us_list_too_long():
    """
    Passing more than 2 elements in the Us list raises ValueError.
    A range needs exactly a lower and an upper bound.
    """
    with pytest.raises(ValueError):
        Stability_Criteria(Us=[0.0, 1.0, 2.0], pHs=7.0)


def test_invalid_pHs_list_too_long():
    """
    Same constraint applies to pHs: more than 2 elements raises ValueError.
    """
    with pytest.raises(ValueError):
        Stability_Criteria(Us=0.0, pHs=[6.0, 7.0, 8.0])


def test_invalid_Us_list_empty():
    """
    An empty list for Us raises ValueError — at least one element is required.
    """
    with pytest.raises(ValueError):
        Stability_Criteria(Us=[], pHs=7.0)


def test_invalid_pHs_list_empty():
    """
    An empty list for pHs raises ValueError — at least one element is required.
    """
    with pytest.raises(ValueError):
        Stability_Criteria(Us=0.0, pHs=[])


# ── 2. Scalar point evaluation ────────────────────────────────────────────────

def test_scalar_stable():
    """
    Simplest stable case: scalar U and pH, all decomposition energies below
    the threshold.  evaluate() should return True.
    """
    sc = Stability_Criteria(Us=0.0, pHs=7.0, decomposition_threshold=0.5)
    assert sc.max_dG_in_region(_all_below()) <= sc.decomposition_threshold


def test_scalar_unstable():
    """
    Simplest unstable case: scalar U and pH, all decomposition energies above
    the threshold.  evaluate() should return False.
    """
    sc = Stability_Criteria(Us=0.0, pHs=7.0, decomposition_threshold=0.5)
    assert sc.max_dG_in_region(_all_above()) > sc.decomposition_threshold


def test_single_point_only_passes():
    """
    With scalar Us and pHs, evaluate() reads exactly ONE cell of decom_G:
      decom_G[U_index, pH_index]

    This test fills the entire array above the threshold, then sets only the
    target cell (U=0 → idx 10, pH=7 → idx 9) below it.  The result must be
    True, confirming that no other cell is examined.
    """
    sc = Stability_Criteria(Us=0.0, pHs=7.0, decomposition_threshold=0.5)
    decom_G = _all_above()
    decom_G[10, 9] = 0.2  # only the target cell is stable
    assert sc.max_dG_in_region(decom_G) <= sc.decomposition_threshold


# ── 3. Threshold behaviour ────────────────────────────────────────────────────

def test_threshold_at_exact_boundary():
    """
    evaluate() uses <=, so a cell whose decom_G equals the threshold exactly
    should still count as stable (True).
    """
    sc = Stability_Criteria(Us=0.0, pHs=7.0, decomposition_threshold=0.5)
    decom_G = _all_above()
    decom_G[10, 9] = 0.5  # exactly at the threshold
    assert sc.max_dG_in_region(decom_G) <= sc.decomposition_threshold


def test_custom_threshold():
    """
    The decomposition_threshold is configurable.  A value of 0.2 should
    accept decom_G = 0.15 (below) but reject 0.25 (above).
    """
    sc = Stability_Criteria(Us=0.0, pHs=7.0, decomposition_threshold=0.2)
    assert sc.max_dG_in_region(np.full(SHAPE, 0.15)) <= sc.decomposition_threshold
    assert sc.max_dG_in_region(np.full(SHAPE, 0.25)) > sc.decomposition_threshold


# ── 4. U-range evaluation ─────────────────────────────────────────────────────

def test_U_range_all_pass():
    """
    When Us is a two-element list [lo, hi], evaluate() checks ALL cells in the
    corresponding U-slice at the given pH column.

    Us=[0.0, 1.0], pHs=7.0
      → U-slice: rows 10 to 15 inclusive  (slice [10:16])
      → pH column: index 9

    All cells in that slice are set below threshold → True.
    """
    sc = Stability_Criteria(Us=[0.0, 1.0], pHs=7.0, decomposition_threshold=0.5)
    decom_G = _all_above()
    decom_G[10:16, 9] = 0.2
    assert sc.max_dG_in_region(decom_G) <= sc.decomposition_threshold


def test_U_range_one_fail():
    """
    Same setup as test_U_range_all_pass, but one cell inside the U-slice is
    left above the threshold.  evaluate() must return False because the
    stability condition requires ALL cells in the slice to pass.
    """
    sc = Stability_Criteria(Us=[0.0, 1.0], pHs=7.0, decomposition_threshold=0.5)
    decom_G = _all_above()
    decom_G[10:16, 9] = 0.2   # fill the slice...
    decom_G[12, 9] = 0.8      # ...then break one cell back above threshold
    assert sc.max_dG_in_region(decom_G) > sc.decomposition_threshold


# ── 5. pH-range evaluation ────────────────────────────────────────────────────

def test_pH_range_all_pass():
    """
    When pHs is a two-element list [lo, hi], evaluate() checks ALL cells in
    the corresponding pH-slice at the given U row.

    Us=0.0, pHs=[6.0, 8.0]
      → U row:    index 10                    (U = 0.0 V)
      → pH-slice: columns 8 to 10 inclusive  (slice [8:11])
                  covering pH = 6, 7, 8

    All cells in that slice are set below threshold → True.
    """
    sc = Stability_Criteria(Us=0.0, pHs=[6.0, 8.0], decomposition_threshold=0.5)
    decom_G = _all_above()
    decom_G[10, 8:11] = 0.2
    assert sc.max_dG_in_region(decom_G) <= sc.decomposition_threshold


def test_pH_range_one_fail():
    """
    Same pH-slice as test_pH_range_all_pass, but one cell inside the slice is
    left above the threshold.  evaluate() must return False.
    """
    sc = Stability_Criteria(Us=0.0, pHs=[6.0, 8.0], decomposition_threshold=0.5)
    decom_G = _all_above()
    decom_G[10, 8:11] = 0.2   # fill the pH-slice...
    decom_G[10, 9] = 0.8      # ...then break one cell (pH=7) back above threshold
    assert sc.max_dG_in_region(decom_G) > sc.decomposition_threshold


# ── 6. 2-D range evaluation ───────────────────────────────────────────────────

def test_both_ranges_all_pass():
    """
    When both Us and pHs are ranges, evaluate() checks the full 2-D sub-array
    at the intersection of the two slices.

    Us=[0.0, 1.0], pHs=[6.0, 8.0]
      → U-slice:  rows    10:16  (U  = 0.0 … 1.0 V)
      → pH-slice: columns  8:11  (pH = 6   … 8  )

    All 6×3 = 18 cells in that block are set below threshold → True.
    """
    sc = Stability_Criteria(Us=[0.0, 1.0], pHs=[6.0, 8.0], decomposition_threshold=0.5)
    decom_G = _all_above()
    decom_G[10:16, 8:11] = 0.2
    assert sc.max_dG_in_region(decom_G) <= sc.decomposition_threshold


def test_both_ranges_one_fail():
    """
    Same 2-D block as test_both_ranges_all_pass, but one interior cell is
    left above the threshold.  evaluate() must return False.
    """
    sc = Stability_Criteria(Us=[0.0, 1.0], pHs=[6.0, 8.0], decomposition_threshold=0.5)
    decom_G = _all_above()
    decom_G[10:16, 8:11] = 0.2   # fill the block...
    decom_G[13, 9] = 0.8         # ...then break one interior cell
    assert sc.max_dG_in_region(decom_G) > sc.decomposition_threshold
