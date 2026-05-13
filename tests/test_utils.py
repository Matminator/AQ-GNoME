"""
Tests for get_simplified_df (utils.py)
========================================

Only get_simplified_df is tested here — it is a pure DataFrame transformation
with no file I/O.  atoms_from_db requires a real ASE database and is excluded.
"""

import pandas as pd
import pytest
from aq_gnome.utils import get_simplified_df


# Columns that get_simplified_df is expected to drop
DROPPED_COLUMNS = [
    "gga_only_pbx_save_id",
    "mixed_pbx_save_id",
    "Data Directory",
    "Is Train",
    "Decomposition Energy Per Atom Relative",
    "Decomposition Energy Per Atom All",
    "Decomposition Energy Per Atom MP",
    "Decomposition Energy Per Atom MP OQMD",
    "Decomposition Energy Per Atom",
    "Corrected Energy",
    "Space Group Number",
    "Uncorrected Energy",
    "Space Group",
    "Density",
    "Volume",
    "Point Group",
    "Formation Energy Per Atom",
]


@pytest.fixture
def full_df():
    """DataFrame containing all columns that should be dropped plus extras."""
    data = {col: [1, 2] for col in DROPPED_COLUMNS}
    data["MaterialId"] = ["m001", "m002"]
    data["Bandgap"] = [0.0, 1.2]
    return pd.DataFrame(data)


def test_simplified_df_drops_expected_columns(full_df):
    result = get_simplified_df(full_df)
    for col in DROPPED_COLUMNS:
        assert col not in result.columns


def test_simplified_df_keeps_other_columns(full_df):
    result = get_simplified_df(full_df)
    assert "MaterialId" in result.columns
    assert "Bandgap" in result.columns


def test_simplified_df_missing_columns_no_error():
    """If a to-be-dropped column is absent, no exception should be raised."""
    df = pd.DataFrame({"MaterialId": ["m001"], "Bandgap": [0.0]})
    result = get_simplified_df(df)
    assert list(result.columns) == ["MaterialId", "Bandgap"]


def test_simplified_df_returns_copy(full_df):
    """The original DataFrame should be unchanged after the call."""
    original_cols = set(full_df.columns)
    get_simplified_df(full_df)
    assert set(full_df.columns) == original_cols


def test_simplified_df_empty_df():
    """An empty DataFrame passes through without error."""
    df = pd.DataFrame(columns=["MaterialId"] + DROPPED_COLUMNS)
    result = get_simplified_df(df)
    assert len(result) == 0
    assert "MaterialId" in result.columns
