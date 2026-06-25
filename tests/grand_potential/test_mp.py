"""Unit tests for run-type filtering in ``aq_gnome.grand_potential._mp`` (no network)."""

import types

import pytest

from aq_gnome.grand_potential._mp import apply_corrections, filter_run_types, _make_compat


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


# ----------------------------------------- apply_corrections: target/run-type handling
#
# Regression for a real bug: when a GNoME target's structure exists in MP only as an r2SCAN
# calculation, the substitution used to swap in that r2SCAN entry and then the pipeline lost it —
# deleted by the GGA run-type filter, or discarded by the mixing scheme as a lone (unanchored)
# r2SCAN entry. The candidate was silently dropped ("did not survive the compatibility filter").

class _FakeEntry:
    """Minimal ComputedStructureEntry stand-in: id, run_type, and a hashable structure key."""

    def __init__(self, entry_id, run_type, structure):
        self.entry_id = entry_id
        self.parameters = {"run_type": run_type}
        self.structure = structure


class _Matcher:
    """StructureMatcher stand-in: two structures match iff their keys are equal."""

    def fit(self, s1, s2):
        return s1 == s2


class _IdentityCompat:
    """No corrections, no drops — like MP2020Compatibility on already-valid GGA entries."""

    def process_entries(self, pool):
        return list(pool)


class _MixingLikeCompat:
    """Toy DFT mixing scheme: an r2SCAN entry survives only if a same-structure GGA entry anchors
    it; a GGA entry that has an r2SCAN twin is collapsed away. Reproduces why a lone r2SCAN target
    (no GGA anchor in the pool) is discarded."""

    def __init__(self):
        self.seen = None

    def process_entries(self, pool):
        self.seen = list(pool)
        r2 = {"r2SCAN", "R2SCAN"}
        gga_structs = {e.structure for e in pool if e.parameters.get("run_type") not in r2}
        r2_structs = {e.structure for e in pool if e.parameters.get("run_type") in r2}
        out = []
        for e in pool:
            if e.parameters.get("run_type") in r2:
                if e.structure in gga_structs:          # anchored -> survives as representative
                    out.append(e)
            elif e.structure not in r2_structs:         # GGA with no r2SCAN twin -> kept
                out.append(e)
        return out


def test_gga_mode_keeps_target_when_mp_twin_is_r2scan_only():
    # GGA mode must NOT substitute the target with an r2SCAN-only MP twin (which the run-type filter
    # then deletes); it keeps the GNoME GGA entry instead. Pre-fix this raised "did not survive".
    target = _FakeEntry("GGA_x", "GGA", "S")
    mp = [_FakeEntry("mp-1-r2SCAN", "r2SCAN", "S"), _FakeEntry("mp-2-GGA", "GGA", "OTHER")]
    _, _, t_corr, mp_sub = apply_corrections(
        mp, [target], [], _IdentityCompat(), "GGA", structure_matcher=_Matcher())
    assert mp_sub is False
    assert str(t_corr.entry_id) == "GGA_x"


def test_mixing_mode_anchors_r2scan_only_twin_with_gnome_gga():
    # GGA+r2SCAN: an r2SCAN-only MP twin gets the GNoME GGA entry fed as the anchor, so exactly one
    # (r2SCAN) target survives instead of being discarded as a lone r2SCAN entry.
    target = _FakeEntry("GGA_x", "GGA", "S")
    mp = [_FakeEntry("mp-1-r2SCAN", "r2SCAN", "S")]
    compat = _MixingLikeCompat()
    _, _, t_corr, mp_sub = apply_corrections(
        mp, [target], [], compat, "GGA+r2SCAN", structure_matcher=_Matcher())
    assert mp_sub is True
    assert t_corr.parameters["run_type"] in {"r2SCAN", "R2SCAN"}
    assert "GGA_x" in [str(e.entry_id) for e in compat.seen]   # the GGA anchor was fed to the scheme


def test_gga_twin_is_substituted_normally():
    # Happy path unchanged: a GGA MP twin still substitutes the target (mp_substituted=True).
    target = _FakeEntry("GGA_x", "GGA", "S")
    mp = [_FakeEntry("mp-1-GGA", "GGA", "S")]
    _, _, t_corr, mp_sub = apply_corrections(
        mp, [target], [], _IdentityCompat(), "GGA", structure_matcher=_Matcher())
    assert mp_sub is True
    assert str(t_corr.entry_id).startswith("MP_GNOME")
