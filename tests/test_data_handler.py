"""
Tests for Data_Handler (data_handler.py)
=========================================

Background
----------
Data_Handler loads CSVs and HDF5 databases on __init__, which requires the
real data/ directory.  To test the filtering logic without real files, every
test here bypasses __init__ by creating a bare instance (object.__new__) and
injecting a synthetic DataFrame directly.  This is safe because the filtering
methods only touch self.modified_df, self.combined_df, and self.N_total_GNoME.

The synthetic_df fixture is defined in conftest.py.  pytest automatically
makes it available to any test function that lists it as a parameter — no
import is needed.  It contains 4 rows:
    m001  [Fe, O]   3D  Bandgap=0.0
    m002  [Fe, O]   3D  Bandgap=1.2
    m003  [Ti, O]   2D  Bandgap=3.0
    m004  [Ni, O]   3D  Bandgap=0.0
"""

import pytest
import pandas as pd
from pathlib import Path
from aq_gnome.data_handler import Data_Handler


# ── Helper ────────────────────────────────────────────────────────────────────

def _make_handler(df: pd.DataFrame) -> Data_Handler:
    """
    Return a Data_Handler with combined_df / modified_df set to df, without
    touching the filesystem.  object.__new__(Data_Handler) creates an instance
    that skips __init__ entirely; we then set the three attributes the
    filtering methods actually use.
    """
    dh = object.__new__(Data_Handler)
    dh.combined_df = df.copy()
    dh.modified_df = df.copy()
    dh.N_total_GNoME = len(df)
    return dh


# ── remove_entries_with_elements ──────────────────────────────────────────────

def test_remove_entries_with_elements_string(synthetic_df):
    """Passing a single element as a string removes all rows containing it."""
    dh = _make_handler(synthetic_df)
    dh.remove_entries_with_elements("Ti")
    result = dh.get_df()
    assert "m003" not in result["MaterialId"].values
    assert len(result) == 3


def test_remove_entries_with_elements_list(synthetic_df):
    """Passing a list removes any row containing any of the listed elements."""
    dh = _make_handler(synthetic_df)
    dh.remove_entries_with_elements(["Ti", "Ni"])
    result = dh.get_df()
    assert set(result["MaterialId"]) == {"m001", "m002"}


def test_remove_entries_with_elements_no_match(synthetic_df):
    """When no row contains the element the DataFrame is unchanged."""
    dh = _make_handler(synthetic_df)
    dh.remove_entries_with_elements("Cu")
    assert len(dh.get_df()) == len(synthetic_df)


# ── remove_entries_without_elements ──────────────────────────────────────────

def test_remove_entries_without_elements_any(synthetic_df):
    """must_contain_all=False keeps rows that contain ANY of the listed elements."""
    dh = _make_handler(synthetic_df)
    dh.remove_entries_without_elements(["Ti"], must_contain_all=False)
    result = dh.get_df()
    assert set(result["MaterialId"]) == {"m003"}


def test_remove_entries_without_elements_all_required(synthetic_df):
    """must_contain_all=True keeps only rows that contain ALL listed elements."""
    dh = _make_handler(synthetic_df)
    dh.remove_entries_without_elements(["Fe", "O"], must_contain_all=True)
    result = dh.get_df()
    assert set(result["MaterialId"]) == {"m001", "m002"}


def test_remove_entries_without_elements_empty_result(synthetic_df):
    """If no row satisfies the requirement the result is an empty DataFrame."""
    dh = _make_handler(synthetic_df)
    dh.remove_entries_without_elements(["Cu"], must_contain_all=False)
    assert len(dh.get_df()) == 0


# ── remove_entries_not_consisting_exclusively_of_elements ────────────────────

def test_remove_entries_not_exclusively(synthetic_df):
    """Only rows whose element set is a subset of the allowed list are kept."""
    dh = _make_handler(synthetic_df)
    dh.remove_entries_not_consisting_exclusively_of_elements(["Fe", "O"])
    result = dh.get_df()
    assert set(result["MaterialId"]) == {"m001", "m002"}


# ── restore_df ────────────────────────────────────────────────────────────────

def test_restore_df_resets_all_filters(synthetic_df):
    """restore_df() brings the DataFrame back to its original state."""
    dh = _make_handler(synthetic_df)
    dh.remove_entries_with_elements(["Ti", "Ni"])
    assert len(dh.get_df()) == 2  # filtered
    dh.restore_df()
    assert len(dh.get_df()) == 4  # back to full


# ── get_df ────────────────────────────────────────────────────────────────────

def test_get_df_returns_copy(synthetic_df):
    """Modifying the returned DataFrame must not affect the internal state."""
    dh = _make_handler(synthetic_df)
    returned = dh.get_df()
    returned["MaterialId"] = "CHANGED"
    assert dh.get_df()["MaterialId"].tolist() == ["m001", "m002", "m003", "m004"]


# ── chaining filters ──────────────────────────────────────────────────────────

def test_chained_filters_compose(synthetic_df):
    """Multiple filter calls reduce the DataFrame incrementally."""
    dh = _make_handler(synthetic_df)
    assert len(dh.get_df()) == 4  # start: 4 rows

    dh.remove_entries_with_elements(["Ni"])
    assert len(dh.get_df()) == 3  # m004 removed

    dh.remove_entries_without_elements(["Fe"], must_contain_all=False)
    assert len(dh.get_df()) == 2  # m003 removed (no Fe)
    assert set(dh.get_df()["MaterialId"]) == {"m001", "m002"}


# ── __init__ path handling ────────────────────────────────────────────────────

def test_init_str_nonexistent_raises_not_a_directory():
    """A str path that does not exist → NotADirectoryError."""
    with pytest.raises(NotADirectoryError):
        Data_Handler(solid_filter=True, gga_only=True,
                     path_to_data_directory="/nonexistent/path/abc")


def test_init_path_object_nonexistent_raises_not_a_directory():
    """A Path object that does not exist → NotADirectoryError (not UnboundLocalError)."""
    with pytest.raises(NotADirectoryError):
        Data_Handler(solid_filter=True, gga_only=True,
                     path_to_data_directory=Path("/nonexistent/path/abc"))


def test_init_path_object_existing_dir_proceeds_past_path_check(tmp_path):
    """A valid Path object passes is_dir() and only fails later at CSV loading
    (FileNotFoundError), confirming no UnboundLocalError in path resolution."""
    with pytest.raises(FileNotFoundError):
        Data_Handler(solid_filter=True, gga_only=True,
                     path_to_data_directory=tmp_path)


def test_init_str_existing_dir_proceeds_past_path_check(tmp_path):
    """Baseline: a valid str path behaves the same — fails at CSV loading only."""
    with pytest.raises(FileNotFoundError):
        Data_Handler(solid_filter=True, gga_only=True,
                     path_to_data_directory=str(tmp_path))
