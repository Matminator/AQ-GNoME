"""
Tests for visualization.py
============================

Only the two pure color-dict functions are tested here.
plot_periodic_table_with_values() is a pure rendering function with no
return value or computable output, so it is not tested.
"""

import pytest
from aq_gnome.visualization import (
    get_col_dict_for_atoms,
    get_col_dict_for_each_list_of_elements,
)


# ── get_col_dict_for_atoms ────────────────────────────────────────────────────

def test_col_dict_discard_assigns_red():
    result = get_col_dict_for_atoms(elements_to_discard=["Fe", "Ni"])
    assert result["Fe"] == "tab:red"
    assert result["Ni"] == "tab:red"


def test_col_dict_include_assigns_green():
    result = get_col_dict_for_atoms(elements_to_include=["O"])
    assert result["O"] == "tab:green"


def test_col_dict_allowed_assigns_blue():
    result = get_col_dict_for_atoms(elements_exclusively_allowed=["Ti"])
    assert result["Ti"] == "tab:blue"


def test_col_dict_empty_input_returns_empty():
    result = get_col_dict_for_atoms()
    assert result == {}


def test_col_dict_discard_overrides_include():
    """Elements listed in both discard and include: discard (red) is applied last."""
    result = get_col_dict_for_atoms(
        elements_to_include=["O"],
        elements_to_discard=["O"],
    )
    assert result["O"] == "tab:red"


# ── get_col_dict_for_each_list_of_elements ────────────────────────────────────

def test_col_dict_for_each_list_assigns_distinct_colors():
    """Two non-overlapping lists get different colors."""
    result = get_col_dict_for_each_list_of_elements([["Fe"], ["Ti"]])
    assert result["Fe"] != result["Ti"]


def test_col_dict_for_each_list_same_element_last_wins():
    """If the same element appears in two lists, the later list's color wins."""
    result_0 = get_col_dict_for_each_list_of_elements([["O"], []])
    result_1 = get_col_dict_for_each_list_of_elements([[], ["O"]])
    assert result_0["O"] != result_1["O"]
