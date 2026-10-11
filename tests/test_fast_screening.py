"""
Tests for fast_screening.py — the vectorised twin of Stable_Entries.

The contract is simple: for the same Data_Handler and Stability_Criteria, the fast path must
return exactly the rows, column layout and rounded values that Stable_Entries.get_stable_df()
returns. The no-data tests pin the pieces (region selection, element filters, grid loading) on
the synthetic fixtures from conftest.py; the requires_data test runs both implementations on a
real subset of GNoME for both functional variants.
"""
import types

import numpy as np
import pandas as pd
import pytest

from aq_gnome.database import AQ_H5Database
from aq_gnome.stability import Stability_Criteria, Stable_Entries
from aq_gnome.fast_screening import (
    load_decomposition_grid, element_membership, element_mask, region_max, screen,
    stable_dataframe,
)

CRITERIA_SHAPES = [
    dict(Us=[1.2, 2.0], pHs=[0, 2], decomposition_threshold=0.5),   # range × range
    dict(Us=[1.2, 2.0], pHs=0, decomposition_threshold=0.2),        # range × single
    dict(Us=0.0, pHs=[2, 5], decomposition_threshold=0.1),          # single × range
    dict(Us=-1.0, pHs=14, decomposition_threshold=0.05),            # single × single
]


def _assert_same_result(fast: pd.DataFrame, reference: pd.DataFrame):
    assert list(fast.columns) == list(reference.columns)
    assert list(fast['MaterialId']) == list(reference['MaterialId'])
    for col in fast.columns:
        if col.startswith('max_dG'):
            assert list(fast[col]) == list(reference[col])


# ── No data required ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("params", CRITERIA_SHAPES)
def test_region_max_matches_reference_loop(params):
    rng = np.random.default_rng(0)
    grid = rng.uniform(-1, 5, size=(50, 31, 19)).astype(np.float32)
    sc = Stability_Criteria(**params)
    expected = np.array([sc.max_dG_in_region(g) for g in grid])
    got = region_max(grid, sc)
    assert got.dtype == np.float64
    np.testing.assert_array_equal(got, expected)


def test_element_mask_semantics(synthetic_df):
    # synthetic_df: m001 Fe-O, m002 Fe-O, m003 Ti-O, m004 Ni-O
    mem = element_membership(synthetic_df)
    assert list(mem.columns) == ['Fe', 'Ni', 'O', 'Ti']
    ids = synthetic_df['MaterialId'].to_numpy()

    assert list(ids[element_mask(mem, exclude=['Ti'])]) == ['m001', 'm002', 'm004']
    assert list(ids[element_mask(mem, require=['Fe', 'O'], require_all=True)]) == ['m001', 'm002']
    assert list(ids[element_mask(mem, require=['Ti', 'Ni'], require_all=False)]) == ['m003', 'm004']
    assert list(ids[element_mask(mem, allowed_only=['Fe', 'O'])]) == ['m001', 'm002']
    # requiring an element absent from the data → empty, like remove_entries_without_elements
    assert element_mask(mem, require=['Fe', 'Xx'], require_all=True).sum() == 0
    # unknown elements in exclude / any-require are simply ignored
    assert element_mask(mem, exclude=['Xx']).all()
    assert list(ids[element_mask(mem, require=['Xx', 'Ni'], require_all=False)]) == ['m004']


@pytest.fixture
def mock_dh(synthetic_df, h5_path_with_data):
    db = AQ_H5Database(h5_path_with_data, mode="r")
    dh = types.SimpleNamespace(gga_only=False, gga_results=db, mixed_results=db)
    dh.get_df = lambda: synthetic_df.copy()
    yield dh
    db.close()


def test_grid_matches_stable_entries_on_synthetic_data(mock_dh):
    df = mock_dh.get_df()
    grid = load_decomposition_grid(mock_dh, df)
    assert grid.shape == (4, 31, 19) and grid.dtype == np.float32
    # entry i has all cells 0.1*(i+1); m002 falls back to GGA ('Not computed' mixed id)
    np.testing.assert_allclose(grid[:, 0, 0], [0.1, 0.2, 0.3, 0.4], rtol=1e-6)

    criteria = [Stability_Criteria(Us=[0, 1], pHs=[0, 7], decomposition_threshold=0.25)]
    mask, maxes = screen(grid, criteria)
    fast = stable_dataframe(df, mask, criteria, maxes)
    reference = Stable_Entries(mock_dh, criteria).get_stable_df()
    assert list(fast['MaterialId']) == ['m001', 'm002']
    _assert_same_result(fast, reference)


# ── Real data ──────────────────────────────────────────────────────────────────

@pytest.mark.requires_data
@pytest.mark.parametrize("gga_only", [False, True])
def test_fast_path_equals_stable_entries_on_real_subset(data_dir, gga_only):
    from aq_gnome import Data_Handler

    dh = Data_Handler(solid_filter=True, gga_only=gga_only, path_to_data_directory=data_dir)
    # ~6k H-containing materials (hydrides, hydroxides, …): some pass any sensible criterion,
    # and a share of them lack a mixed-functional grid, exercising the GGA fallback.
    dh.remove_entries_without_elements(['H'], must_contain_all=True)
    df = dh.get_df()
    assert len(df) > 1000, "subset unexpectedly small — test would not be meaningful"
    grid = load_decomposition_grid(dh, df)
    assert grid.shape == (len(df), 31, 19)

    for i, params in enumerate((CRITERIA_SHAPES[0], CRITERIA_SHAPES[3])):
        criteria = [Stability_Criteria(**params),
                    Stability_Criteria(Us=[0, 0.8], pHs=[6, 8], decomposition_threshold=0.3)]
        mask, maxes = screen(grid, criteria)
        fast = stable_dataframe(df, mask, criteria, maxes)
        reference = Stable_Entries(dh, criteria).get_stable_df()
        if i == 0:
            assert len(fast) > 0, "lenient criterion should keep some materials"
        _assert_same_result(fast, reference)
