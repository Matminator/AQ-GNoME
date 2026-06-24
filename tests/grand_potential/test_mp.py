"""Unit tests for run-type filtering in ``aq_gnome.grand_potential._mp`` (no network)."""

import types

import pytest

from aq_gnome.grand_potential._mp import filter_run_types, _make_compat


def _entry(run_type):
    """Minimal stand-in for a ComputedStructureEntry (only .parameters is used)."""
    return types.SimpleNamespace(parameters={"run_type": run_type})


def test_gga_keeps_gga_and_ggau_drops_r2scan():
    entries = [_entry("GGA"), _entry("GGA+U"), _entry("r2SCAN")]
    out = filter_run_types(entries, "GGA")
    assert [e.parameters["run_type"] for e in out] == ["GGA", "GGA+U"]


def test_mixing_keeps_all_three():
    entries = [_entry("GGA"), _entry("GGA+U"), _entry("r2SCAN")]
    out = filter_run_types(entries, "GGA+r2SCAN")
    assert len(out) == 3


def test_drops_unknown_or_missing_run_type():
    entries = [_entry("GGA"), _entry("PBEsol"), _entry(None)]
    assert [e.parameters["run_type"] for e in filter_run_types(entries, "GGA")] == ["GGA"]


@pytest.mark.parametrize("mixing", ["GGA", "GGA+r2SCAN"])
def test_empty_input_returns_empty(mixing):
    assert filter_run_types([], mixing) == []


# ----------------------------------------------------------------- _make_compat

def test_make_compat_gga_returns_mp2020():
    from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
    assert isinstance(_make_compat("GGA"), MaterialsProject2020Compatibility)


def test_make_compat_mixed_returns_mixing_scheme():
    from pymatgen.entries.mixing_scheme import MaterialsProjectDFTMixingScheme
    assert isinstance(_make_compat("GGA+r2SCAN"), MaterialsProjectDFTMixingScheme)


@pytest.mark.parametrize("bad", ["gga", "GGA+R2SCAN", "r2SCAN", "foo", ""])
def test_make_compat_rejects_unknown_mixing(bad):
    with pytest.raises(ValueError, match="mixing must be one of"):
        _make_compat(bad)
