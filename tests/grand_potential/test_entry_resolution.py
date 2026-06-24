"""No-network tests for the analyzer's MaterialId-driven entry resolution + competitor enumeration.

A fake GNoME db returns the pbx id itself as the "entry", so tests can assert exactly which entries
the analyzer would load. The MP cache is never touched here (compute_one is not called).
"""

import numpy as np
import pandas as pd
import pytest

from aq_gnome.grand_potential import GrandPotentialAnalyzer, ResultsCache


class _FakeDB:
    def get_entry(self, pbx_id):
        return pbx_id  # stand-in for the ComputedStructureEntry


def _df():
    return pd.DataFrame({
        "MaterialId":           ["m1", "m2", "m3", "m4", "m5"],
        "Elements":             [["Co", "O"], ["Co", "O"], ["Co", "Br", "O"], ["Ni", "O"], ["Co", "O"]],
        "gga_only_pbx_save_id": ["GGA_m1", "GGA_m2", "GGA_m3", "GGA_m4", "GGA_m5"],
        "mixed_pbx_save_id":    ["r2S_m1", "Not computed", "GGA_m3", "r2S_m4", "r2S_m5"],
        "Reduced Formula":      ["CoO", "CoO2", "CoBrO3", "NiO", "Co3O4"],
    })


def _analyzer(mixing="GGA", include_gnome_competitors=False):
    return GrandPotentialAnalyzer(
        cache=None,                      # not used unless compute_one runs
        gnome_db=_FakeDB(),
        gnome_df=_df(),
        include_gnome_competitors=include_gnome_competitors,
        mixing=mixing,
    )


# ---------------------------------------------------------------- _resolve_entries

def test_gga_mode_loads_only_gga():
    az = _analyzer(mixing="GGA")
    assert az._resolve_entries(az._df.loc["m1"]) == ["GGA_m1"]  # r2S exists but GGA mode ignores it


def test_r2scan_mode_loads_both_when_available():
    az = _analyzer(mixing="GGA+r2SCAN")
    assert az._resolve_entries(az._df.loc["m1"]) == ["GGA_m1", "r2S_m1"]


def test_r2scan_mode_gga_only_when_not_computed():
    az = _analyzer(mixing="GGA+r2SCAN")
    assert az._resolve_entries(az._df.loc["m2"]) == ["GGA_m2"]  # mixed = 'Not computed'


def test_r2scan_mode_gga_only_when_mixed_is_gga():
    az = _analyzer(mixing="GGA+r2SCAN")
    assert az._resolve_entries(az._df.loc["m3"]) == ["GGA_m3"]  # mixed id is GGA_, not r2S_


# ------------------------------------------------- _enumerate_competitor_entries

def test_enumerate_disabled_returns_empty():
    az = _analyzer(include_gnome_competitors=False)
    assert az._enumerate_competitor_entries({"Co", "O"}, "m1") == []


def test_enumerate_returns_subset_excluding_target():
    az = _analyzer(mixing="GGA", include_gnome_competitors=True)
    # In {Co,O}: m2 and m5 are subsets; m1 is the target (excluded); m3 (Co,Br,O) and m4 (Ni,O) are not.
    assert az._enumerate_competitor_entries({"Co", "O"}, "m1") == ["GGA_m2", "GGA_m5"]


def test_enumerate_loads_r2scan_competitors_in_mixed_mode():
    az = _analyzer(mixing="GGA+r2SCAN", include_gnome_competitors=True)
    # m2 has no r2SCAN (GGA only); m5 has r2SCAN (GGA + r2SCAN).
    assert az._enumerate_competitor_entries({"Co", "O"}, "m1") == ["GGA_m2", "GGA_m5", "r2S_m5"]


# ------------------------------------------------- from_results_cache (adopt config)

def test_from_results_cache_adopts_stored_config(tmp_path):
    rc = ResultsCache(tmp_path / "r.h5", force=True)
    rc.bind({
        "mixing": "GGA+r2SCAN", "P_O2": 0.21, "si_reference": "SiO2",
        "include_gnome_competitors": True, "T_values": [0.0, 500.0, 1000.0],
    })
    az = GrandPotentialAnalyzer.from_results_cache(rc, cache=None, gnome_db=_FakeDB(), gnome_df=_df())
    assert az.mixing == "GGA+r2SCAN"
    assert az.P_O2 == 0.21
    assert az.si_reference == "SiO2"
    assert az.include_gnome_competitors is True
    assert list(az.T_values) == [0.0, 500.0, 1000.0]
    assert az.results_cache is rc
    rc.close()


def test_from_results_cache_empty_store_raises(tmp_path):
    rc = ResultsCache(tmp_path / "r.h5", force=True)  # never bound -> no config
    with pytest.raises(ValueError, match="no config yet"):
        GrandPotentialAnalyzer.from_results_cache(rc, cache=None, gnome_db=_FakeDB(), gnome_df=_df())
    rc.close()


def test_from_results_cache_uses_exact_T_grid(tmp_path):
    # Must adopt the store's exact /T_grid, not the 6-dp-rounded config T_values, so curves and
    # decomp transitions added via the attach workflow line up with what's already stored.
    rc = ResultsCache(tmp_path / "r.h5", force=True)
    grid = list(np.linspace(0.0, 1500.0, 7))  # non-round decimals
    rc.bind({"mixing": "GGA", "P_O2": 1.0, "si_reference": None,
             "include_gnome_competitors": False, "T_values": grid})
    az = GrandPotentialAnalyzer.from_results_cache(rc, cache=None, gnome_db=_FakeDB(), gnome_df=_df())
    assert np.array_equal(np.asarray(az.T_values), np.asarray(rc.T_grid))
    rc.close()
