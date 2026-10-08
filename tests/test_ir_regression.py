"""
Regression tests for the Ir case study screening pipeline.

Reproduces the selection_of_Ir_containing_compounds.ipynb pipeline up to (but not
including) the Materials Project database filter, which queries a live external API
and is not reproducible across MP database releases.

Tests are skipped automatically when data/ is absent or when snapshot files have not
been generated (run the save-snapshot cells in the Ir notebook first).

Runtime: ~30 s (Stable_Entries iterates over ~41 000 materials).
"""
import pytest

from aq_gnome import Data_Handler, Stable_Entries, Stability_Criteria
from tests._regression_helpers import assert_pipeline_result

# ── Expected counts at each pipeline checkpoint ────────────────────────────────
EXPECTED_AFTER_EXCLUDE = 245442  # after remove_entries_with_elements
EXPECTED_AFTER_REQUIRE = 41690   # after remove_entries_without_elements(['Ir'])
EXPECTED_AFTER_HHI = 53         # after NSites / HHI filter

# ── Element lists (identical to notebook cell 331325f9) ────────────────────────
_RADIOACTIVE = [
    'Tc', 'Ra', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn',
    'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og', 'Pm', 'Ac', 'Th', 'Pa', 'U', 'Np',
    'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Po', 'At', 'Rn',
]
_TOXIC = ['Tl', 'Pb', 'As', 'Cd', 'Hg']
_REACTIVE_WITH_WATER = ['Li', 'Na', 'K', 'Rb', 'Cs', 'Fr', 'Ca', 'Ba']
ELEMENTS_TO_EXCLUDE = _RADIOACTIVE + _TOXIC + _REACTIVE_WITH_WATER + [
    'F', 'Se', 'Te', 'B', 'P', 'S', 'Cl', 'C',
]
ELEMENTS_MUST_INCLUDE = ['Ir']


# ── Pipeline fixture ───────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def ir_results(data_dir):
    """Run the full Ir pipeline once and cache results for all tests in this module."""
    dh = Data_Handler(solid_filter=False, gga_only=False, path_to_data_directory=data_dir)
    dh.remove_entries_with_elements(ELEMENTS_TO_EXCLUDE)
    n_after_exclude = len(dh.get_df())
    dh.remove_entries_without_elements(ELEMENTS_MUST_INCLUDE, must_contain_all=True)
    n_after_require = len(dh.get_df())

    scs = [Stability_Criteria(pHs=[0], Us=[1.2, 2], decomposition_threshold=0.2)]
    df_stable = Stable_Entries(dh, scs).get_stable_df()

    df_filtered = (
        df_stable[df_stable['NSites'] < 40]
        [lambda d: d['average_HHI_P_excluding_O_H'] < 5000]
        [lambda d: d['max_HHI_P'] < 9500]
        .sort_values('Bandgap')
        .reset_index(drop=True)
    )

    return n_after_exclude, n_after_require, df_stable, df_filtered


# ── Tests ──────────────────────────────────────────────────────────────────────

@pytest.mark.requires_data
def test_ir_element_filter_counts(ir_results):
    """Element filter steps must retain the expected row counts."""
    n_excl, n_req, _, _ = ir_results
    assert n_excl == EXPECTED_AFTER_EXCLUDE, (
        f"Ir — after remove_entries_with_elements: {n_excl} rows, expected {EXPECTED_AFTER_EXCLUDE}"
    )
    assert n_req == EXPECTED_AFTER_REQUIRE, (
        f"Ir — after remove_entries_without_elements(['Ir']): {n_req} rows, expected {EXPECTED_AFTER_REQUIRE}"
    )


@pytest.mark.requires_data
def test_ir_stable_entries(ir_results):
    """Stable Ir entries must exactly match Ir_candidates.csv (73 rows)."""
    _, _, df_stable, _ = ir_results
    assert_pipeline_result(df_stable, "Ir_candidates", "Ir — stable entries")


@pytest.mark.requires_data
def test_ir_hhi_filter_count(ir_results):
    """After NSites / HHI filter: must retain exactly 53 candidates."""
    _, _, _, df_filtered = ir_results
    assert len(df_filtered) == EXPECTED_AFTER_HHI, (
        f"Ir — after HHI/NSites filter: {len(df_filtered)} rows, expected {EXPECTED_AFTER_HHI}"
    )
