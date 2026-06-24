"""Unit tests for the pure helper functions in ``aq_gnome.grand_potential._cache``.

No network, no database — just the chemsys / normalisation / chunking helpers.
"""

import pandas as pd
import pytest
from pymatgen.core import Composition

from aq_gnome.grand_potential._cache import _subsystems, _element_sets, _chunks


# ----------------------------------------------------------------- _subsystems

def test_subsystems_two_elements():
    assert set(_subsystems(["Co", "O"])) == {"Co", "O", "Co-O"}


def test_subsystems_single_element():
    assert _subsystems(["O"]) == ["O"]


def test_subsystems_empty():
    assert _subsystems([]) == []


@pytest.mark.parametrize("elements", [
    ["O"],
    ["Co", "O"],
    ["Co", "O", "Si"],
    ["Br", "Co", "O", "Si"],
    ["Co", "Mo", "O", "Si", "Zn"],
])
def test_subsystems_count_is_2n_minus_1(elements):
    assert len(_subsystems(elements)) == 2 ** len(set(elements)) - 1


def test_subsystems_are_unique():
    subs = _subsystems(["Br", "Co", "O", "Si"])
    assert len(subs) == len(set(subs))


def test_subsystems_order_independent():
    assert set(_subsystems(["O", "Co", "Br"])) == set(_subsystems(["Br", "O", "Co"]))


def test_subsystems_each_chemsys_is_alphabetical():
    # Every multi-element chemsys must be sorted alphabetically (e.g. 'Co-O', never 'O-Co').
    for chemsys in _subsystems(["O", "Co", "Br"]):
        parts = chemsys.split("-")
        assert parts == sorted(parts)


def test_subsystems_dedups_repeated_input():
    assert set(_subsystems(["O", "O", "Co"])) == {"O", "Co", "Co-O"}


def test_subsystems_accepts_set_and_tuple():
    assert set(_subsystems({"Co", "O"})) == set(_subsystems(("Co", "O")))


def test_subsystems_match_pymatgen_chemical_system():
    # The keys _subsystems produces must equal Composition.chemical_system, since that is the
    # form entries are stored and looked up under in the cache. If these ever diverge, reads
    # silently return nothing.
    subs = set(_subsystems(["Br", "Co", "O", "Si"]))
    for formula in ["O2", "CoO2", "Co3O4", "CoBr2", "SiO2", "CoBrO3"]:
        assert Composition(formula).chemical_system in subs


# ---------------------------------------------------------------- _element_sets

def test_element_sets_from_dataframe():
    df = pd.DataFrame({"Elements": [["Fe", "O"], ["Ti", "O", "O"]]})
    assert _element_sets(df) == [{"Fe", "O"}, {"Ti", "O"}]


def test_element_sets_from_list_of_lists():
    assert _element_sets([["Fe", "O"], ["Ti", "O"]]) == [{"Fe", "O"}, {"Ti", "O"}]


def test_element_sets_from_list_of_tuples_and_sets():
    assert _element_sets([("Fe", "O"), {"Ti", "O"}]) == [{"Fe", "O"}, {"Ti", "O"}]


def test_element_sets_preserves_row_count_and_order():
    assert _element_sets([["A"], ["B"], ["C"]]) == [{"A"}, {"B"}, {"C"}]


def test_element_sets_returns_sets():
    out = _element_sets([["Fe", "O"]])
    assert all(isinstance(s, set) for s in out)


def test_element_sets_dataframe_and_list_agree():
    rows = [["Fe", "O"], ["Ti", "O"]]
    df = pd.DataFrame({"Elements": rows})
    assert _element_sets(df) == _element_sets(rows)


def test_element_sets_empty_inputs():
    assert _element_sets([]) == []
    assert _element_sets(pd.DataFrame({"Elements": []})) == []


# --------------------------------------------------------------------- _chunks

def test_chunks_even_split():
    assert list(_chunks([1, 2, 3, 4], 2)) == [[1, 2], [3, 4]]


def test_chunks_with_remainder():
    assert list(_chunks([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]


def test_chunks_size_larger_than_seq():
    assert list(_chunks([1, 2, 3], 10)) == [[1, 2, 3]]


def test_chunks_size_one():
    assert list(_chunks([1, 2, 3], 1)) == [[1], [2], [3]]


def test_chunks_empty():
    assert list(_chunks([], 3)) == []


@pytest.mark.parametrize("n,size", [(10, 3), (100, 7), (5, 5), (5, 1), (1, 4)])
def test_chunks_reconstructs_original_without_loss(n, size):
    seq = list(range(n))
    chunks = list(_chunks(seq, size))
    # No element lost or duplicated, order preserved.
    assert [x for chunk in chunks for x in chunk] == seq
    # No chunk exceeds the requested size.
    assert all(len(c) <= size for c in chunks)
    # Only the last chunk may be short.
    assert all(len(c) == size for c in chunks[:-1])
