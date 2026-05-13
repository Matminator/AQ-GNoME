"""
Tests for Compound_HHI_scores (HHI_scoring.py)
================================================

Background
----------
Compound_HHI_scores loads a CSV of per-element HHI values and exposes methods
to compute weighted-average and max HHI scores for a material composition.

All tests here use a minimal 4-row fixture CSV written to tmp_path so that no
real data/ directory is needed.

Fixture HHI values (used for manual verification):
    Element   HHI_P   HHI_R
    Fe        1000    500
    O         100     50
    Ti        2000    800
    Ir        9000    9000
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from aq_gnome.HHI_scoring import Compound_HHI_scores


HHI_DATA = pd.DataFrame({
    "Element": ["Fe", "O", "Ti", "Ir"],
    "HHI_P":   [1000, 100, 2000, 9000],
    "HHI_R":   [500,   50,  800, 9000],
})


@pytest.fixture
def hhi_csv(tmp_path):
    """Write the fixture HHI table to a temporary CSV and return the directory."""
    csv_path = tmp_path / "HHI_values.csv"
    HHI_DATA.to_csv(csv_path, index=False)
    return tmp_path


@pytest.fixture
def scorer(hhi_csv):
    return Compound_HHI_scores(path_to_data_folder=hhi_csv)


# ── Initialisation ────────────────────────────────────────────────────────────

def test_init_loads_csv(scorer):
    """HHI_df should be populated after __init__ with a custom path."""
    assert len(scorer.HHI_df) == 4
    assert set(scorer.HHI_df["Element"]) == {"Fe", "O", "Ti", "Ir"}


# ── _compute_HHI_score ────────────────────────────────────────────────────────

def test_compute_HHI_simple_binary(scorer):
    """FeO: 1 Fe (HHI_P=1000) + 1 O (HHI_P=100) → average = 550."""
    element_dict = {"Fe": 1, "O": 1}
    result = scorer._compute_HHI_score(element_dict, "P", exclude_elements=[])
    assert result == 550


def test_compute_HHI_weighted(scorer):
    """Fe2O3: 2 Fe (1000) + 3 O (100) → (2000+300)/5 = 460."""
    element_dict = {"Fe": 2, "O": 3}
    result = scorer._compute_HHI_score(element_dict, "P", exclude_elements=[])
    assert result == 460


def test_compute_HHI_exclude_elements(scorer):
    """Excluding O from FeO leaves only Fe: average = 1000."""
    element_dict = {"Fe": 1, "O": 1}
    result = scorer._compute_HHI_score(element_dict, "P", exclude_elements=["O"])
    assert result == 1000


def test_compute_HHI_all_excluded_returns_zero(scorer):
    """When all elements are excluded the function returns 0."""
    element_dict = {"Fe": 1}
    result = scorer._compute_HHI_score(element_dict, "P", exclude_elements=["Fe"])
    assert result == 0


# ── get_HHI_score ─────────────────────────────────────────────────────────────

def test_get_HHI_score_from_composition(scorer):
    """End-to-end: FeO row should give 550 for HHI_P."""
    row = pd.Series({"Composition": "FeO"})
    assert scorer.get_HHI_score(row, HHI_type="P") == 550


# ── get_max_HHI_score ─────────────────────────────────────────────────────────

def test_get_max_HHI_score_returns_highest(scorer):
    """FeO: max(HHI_P for Fe=1000, O=100) → 1000."""
    row = pd.Series({"Composition": "FeO"})
    assert scorer.get_max_HHI_score(row, HHI_type="P") == 1000


# ── get_HHI_dummys ────────────────────────────────────────────────────────────

def test_get_HHI_dummys_IrO2(scorer):
    """IrO2: (1*9000 + 2*100) / 3 = 3066 (rounded)."""
    expected = int(np.round((9000 + 200) / 3, 0))
    assert scorer.get_HHI_dummys("IrO2", HHI_type="P") == expected


def test_get_HHI_dummys_TiO2(scorer):
    """TiO2: (1*2000 + 2*100) / 3 = 733 (rounded)."""
    expected = int(np.round((2000 + 200) / 3, 0))
    assert scorer.get_HHI_dummys("TiO2", HHI_type="P") == expected


def test_get_HHI_dummys_unknown_raises(scorer):
    with pytest.raises(ValueError):
        scorer.get_HHI_dummys("UnknownO2")
