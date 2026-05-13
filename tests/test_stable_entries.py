"""
Tests for Stable_Entries (stability.py)
========================================

Background
----------
Stable_Entries.find_stable_entries() iterates over a DataFrame, reads a
decomposition-energy array from AQ_H5Database for each row, applies all
Stability_Criteria, and records which materials pass.

In the synthetic setup (conftest.py):
    GGA_m001 → all cells = 0.1
    GGA_m002 → all cells = 0.2
    GGA_m003 → all cells = 0.3
    GGA_m004 → all cells = 0.4

A single Stability_Criteria with threshold T passes entry i iff 0.1*(i+1) <= T.

The tests inject a minimal fake Data_Handler via types.SimpleNamespace.
SimpleNamespace is a lightweight built-in object where you can attach any
attributes you like: `ns = types.SimpleNamespace(x=1)` gives `ns.x == 1`.
It is used here to create a fake Data_Handler that has the same attributes
Stable_Entries reads — without loading any real CSV or HDF5 files.

Both synthetic_df and h5_path_with_data are defined in conftest.py and are
automatically injected by pytest when listed as test parameters.
"""

import types
import numpy as np
import pytest

from aq_gnome.database import AQ_H5Database
from aq_gnome.stability import Stability_Criteria, Stable_Entries


# ── Fixtures: fake Data_Handler ───────────────────────────────────────────────

@pytest.fixture
def mock_dh(synthetic_df, h5_path_with_data):
    """
    Fake Data_Handler backed by the synthetic HDF5 file.
    Both gga_results and mixed_results point to the same file; the fallback
    logic (gga vs mixed) is tested separately in test_get_decom_G_*.
    """
    db = AQ_H5Database(h5_path_with_data, mode="r")
    dh = types.SimpleNamespace(
        gga_only=False,
        gga_results=db,
        mixed_results=db,
    )
    dh.get_df = lambda: synthetic_df.copy()
    yield dh
    db.close()


@pytest.fixture
def mock_dh_gga_only(synthetic_df, h5_path_with_data):
    """Fake Data_Handler with gga_only=True (mixed_results is None)."""
    db = AQ_H5Database(h5_path_with_data, mode="r")
    dh = types.SimpleNamespace(
        gga_only=True,
        gga_results=db,
        mixed_results=None,
    )
    dh.get_df = lambda: synthetic_df.copy()
    yield dh
    db.close()


# ── find_stable_entries ───────────────────────────────────────────────────────

def test_find_stable_entries_all_pass(mock_dh):
    """With threshold=0.5 all four entries (max dG ≤ 0.4) should be stable."""
    sc = Stability_Criteria(Us=0.0, pHs=7.0, decomposition_threshold=0.5)
    se = Stable_Entries(mock_dh, sc)
    se.find_stable_entries()
    assert set(se.ids_of_stable_entries) == {"m001", "m002", "m003", "m004"}


def test_find_stable_entries_none_pass(mock_dh):
    """With threshold=0 no entry survives (all max dG > 0)."""
    sc = Stability_Criteria(Us=0.0, pHs=7.0, decomposition_threshold=0.0)
    se = Stable_Entries(mock_dh, sc)
    se.find_stable_entries()
    assert se.ids_of_stable_entries == []


def test_find_stable_entries_some_pass(mock_dh):
    """threshold=0.15 keeps only m001 (max dG = 0.1 ≤ 0.15)."""
    sc = Stability_Criteria(Us=0.0, pHs=7.0, decomposition_threshold=0.15)
    se = Stable_Entries(mock_dh, sc)
    se.find_stable_entries()
    assert se.ids_of_stable_entries == ["m001"]


def test_find_stable_entries_multiple_criteria(mock_dh):
    """Both criteria must pass; the stricter threshold governs the result."""
    sc_loose  = Stability_Criteria(Us=0.0, pHs=7.0, decomposition_threshold=0.5)
    sc_strict = Stability_Criteria(Us=1.0, pHs=7.0, decomposition_threshold=0.15)
    se = Stable_Entries(mock_dh, [sc_loose, sc_strict])
    se.find_stable_entries()
    assert se.ids_of_stable_entries == ["m001"]


# ── get_stable_df ─────────────────────────────────────────────────────────────

def test_get_stable_df_has_max_dG_column(mock_dh):
    """The returned DataFrame must include the criterion's col_name column."""
    sc = Stability_Criteria(Us=0.0, pHs=7.0, decomposition_threshold=0.5)
    se = Stable_Entries(mock_dh, sc)
    df = se.get_stable_df()
    assert sc.col_name in df.columns


def test_get_stable_df_max_dG_rounded_to_2dp(mock_dh):
    """max_dG values must have at most 2 decimal places."""
    sc = Stability_Criteria(Us=0.0, pHs=7.0, decomposition_threshold=0.5)
    se = Stable_Entries(mock_dh, sc)
    df = se.get_stable_df()
    for val in df[sc.col_name]:
        assert round(val, 2) == val


def test_get_stable_df_is_copy(mock_dh):
    """Mutating the returned DataFrame must not affect the internal cache."""
    sc = Stability_Criteria(Us=0.0, pHs=7.0, decomposition_threshold=0.5)
    se = Stable_Entries(mock_dh, sc)
    df1 = se.get_stable_df()
    df1["MaterialId"] = "CHANGED"
    df2 = se.get_stable_df()
    assert (df2["MaterialId"] != "CHANGED").all()


def test_get_stable_df_cached(mock_dh, monkeypatch):
    """A second call to get_stable_df must not re-run find_stable_entries."""
    sc = Stability_Criteria(Us=0.0, pHs=7.0, decomposition_threshold=0.5)
    se = Stable_Entries(mock_dh, sc)
    se.get_stable_df()  # populates cache

    call_count = {"n": 0}
    original = se.find_stable_entries
    def counted():
        call_count["n"] += 1
        return original()
    monkeypatch.setattr(se, "find_stable_entries", counted)

    se.get_stable_df()  # second call — should NOT invoke find_stable_entries
    assert call_count["n"] == 0


# ── get_decom_G ───────────────────────────────────────────────────────────────

def test_get_decom_G_falls_back_to_gga(mock_dh, synthetic_df):
    """For m002 (mixed_pbx_save_id='Not computed') the GGA database is used."""
    sc = Stability_Criteria(Us=0.0, pHs=7.0, decomposition_threshold=0.5)
    se = Stable_Entries(mock_dh, sc)
    row = synthetic_df[synthetic_df["MaterialId"] == "m002"].iloc[0]
    decom_G = se.get_decom_G(row)
    assert np.allclose(decom_G, 0.2)


def test_get_decom_G_gga_only_mode(mock_dh_gga_only, synthetic_df):
    """In gga_only mode the GGA database is always used."""
    sc = Stability_Criteria(Us=0.0, pHs=7.0, decomposition_threshold=0.5)
    se = Stable_Entries(mock_dh_gga_only, sc)
    row = synthetic_df[synthetic_df["MaterialId"] == "m001"].iloc[0]
    decom_G = se.get_decom_G(row)
    assert np.allclose(decom_G, 0.1)
