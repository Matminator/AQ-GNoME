"""
Regression tests for the CO2RR case study screening pipeline.

Reproduces the finding_CO2RR_candidates.ipynb pipeline up to (but not including) the
Materials Project database filter, which queries a live external API and is not
reproducible across MP database releases.

Tests are skipped automatically when data/ is absent or when snapshot files have not
been generated (run the save-snapshot cells in the CO2RR notebook first).

Runtime: ~3 min (Stable_Entries iterates over ~185 000 materials).
"""
import numpy as np
import pytest
from pymatgen.core import Composition

from aq_gnome import Data_Handler, Stable_Entries, Stability_Criteria
from tests._regression_helpers import assert_pipeline_result

# ── Expected counts at each pipeline checkpoint ────────────────────────────────
EXPECTED_AFTER_EXCLUDE = 245442   # after remove_entries_with_elements
EXPECTED_AFTER_REQUIRE = 185003   # after remove_entries_without_elements(binding els)
EXPECTED_STABLE = 3443            # after Stable_Entries
EXPECTED_BALANCED = 639           # after is_balanced_groups filter

# ── Element lists (identical to notebook cell 331325f9) ────────────────────────
_RADIOACTIVE = [
    'Tc', 'Ra', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn',
    'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og', 'Pm', 'Ac', 'Th', 'Pa', 'U', 'Np',
    'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Po', 'At', 'Rn',
]
_TOXIC = ['Tl', 'Pb', 'As', 'Cd', 'Hg']
_REACTIVE_WITH_WATER = ['Li', 'Na', 'K', 'Rb', 'Cs', 'Fr', 'Ca', 'Ba']
ELEMENTS_TO_EXCLUDE = _RADIOACTIVE + _TOXIC + _REACTIVE_WITH_WATER + [
    'P', 'F', 'Se', 'Te', 'B', 'C', 'S', 'Cl',
]

STRONG_BINDING = ['Ti', 'Ni', 'Pd', 'Pt', 'Ru', 'Rh', 'Ir', 'Fe']
WEAK_BINDING = ['Zn', 'Ag', 'Au', 'Cd', 'In', 'Sn', 'Hg', 'Tl', 'Pb']


def _is_balanced_groups(composition_str, group1, group2, fractional_limit):
    """Return True if composition has a balanced atom count between group1 and group2."""
    comp = Composition(composition_str)
    counts = comp.get_el_amt_dict()
    n1 = sum(counts.get(e, 0) for e in group1)
    n2 = sum(counts.get(e, 0) for e in group2)
    if n1 == 0 or n2 == 0:
        return False
    if np.isclose(fractional_limit, 0):
        return n1 == n2
    return abs(n1 - n2) / (n1 + n2) <= fractional_limit


# ── Pipeline fixture ───────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def co2rr_results(data_dir):
    """Run the full CO2RR pipeline once and cache results for all tests in this module."""
    dh = Data_Handler(solid_filter=True, gga_only=False, path_to_data_directory=data_dir)
    dh.remove_entries_with_elements(ELEMENTS_TO_EXCLUDE)
    n_after_exclude = len(dh.get_df())
    dh.remove_entries_without_elements(STRONG_BINDING + WEAK_BINDING, must_contain_all=False)
    n_after_require = len(dh.get_df())

    scs = [Stability_Criteria(pHs=7, Us=[-4.13700000e-01, -2], decomposition_threshold=0.5)]
    df_stable = Stable_Entries(dh, scs).get_stable_df()

    df_balanced = df_stable[df_stable['Composition'].apply(
        lambda x: _is_balanced_groups(x, STRONG_BINDING, WEAK_BINDING, 0.5)
    )]
    df_disorder = df_balanced[df_balanced['Disorder Probability'] < 0.5]

    return n_after_exclude, n_after_require, df_stable, df_balanced, df_disorder


# ── Tests ──────────────────────────────────────────────────────────────────────

@pytest.mark.requires_data
def test_co2rr_element_filter_counts(co2rr_results):
    """Element filter steps must retain the expected row counts."""
    n_excl, n_req, _, _, _ = co2rr_results
    assert n_excl == EXPECTED_AFTER_EXCLUDE, (
        f"CO2RR — after remove_entries_with_elements: {n_excl} rows, expected {EXPECTED_AFTER_EXCLUDE}"
    )
    assert n_req == EXPECTED_AFTER_REQUIRE, (
        f"CO2RR — after remove_entries_without_elements(binding els): {n_req} rows, expected {EXPECTED_AFTER_REQUIRE}"
    )


@pytest.mark.requires_data
def test_co2rr_stable_count(co2rr_results):
    """Stable entries count must be 3443."""
    _, _, df_stable, _, _ = co2rr_results
    assert len(df_stable) == EXPECTED_STABLE, (
        f"CO2RR — stable entries: {len(df_stable)} rows, expected {EXPECTED_STABLE}"
    )


@pytest.mark.requires_data
def test_co2rr_balanced_count(co2rr_results):
    """After is_balanced_groups filter: must retain 639 candidates."""
    _, _, _, df_balanced, _ = co2rr_results
    assert len(df_balanced) == EXPECTED_BALANCED, (
        f"CO2RR — after balanced-groups filter: {len(df_balanced)} rows, expected {EXPECTED_BALANCED}"
    )


@pytest.mark.requires_data
def test_co2rr_final_candidates(co2rr_results):
    """Final CO2RR candidates must exactly match CO2RR_candidates.csv (55 rows)."""
    _, _, _, _, df_disorder = co2rr_results
    assert_pipeline_result(df_disorder, "CO2RR_candidates", "CO2RR — final candidates")
