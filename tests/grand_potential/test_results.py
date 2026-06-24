"""No-network tests for ResultsCache (aq_gnome.grand_potential._results)."""

import numpy as np
import pytest

from aq_gnome.grand_potential import CurveResult, ResultsCache, list_result_stores
from aq_gnome.grand_potential._results import (
    _decomp_to_transitions, _transitions_to_per_T, _config_hash,
)

T = [0.0, 100.0, 200.0, 300.0]


def _config(P_O2=1.0):
    return {
        "mixing": "GGA",
        "P_O2": P_O2,
        "si_reference": None,
        "include_gnome_competitors": False,
        "T_values": T,
    }


def _result(E, decomp=None, mp_sub=False):
    return CurveResult(np.array(T), np.array(E, dtype=float), decomp or [], mp_sub)


def _new_store(path, config=None):
    """Create + bind a fresh store (creation is force-gated)."""
    rc = ResultsCache(path, force=True)
    rc.bind(config or _config())
    return rc


# --------------------------------------------------------------- decomp helpers

def test_decomp_to_transitions_collapses_runs():
    decomp = [{"A": 1}, {"A": 0.5, "B": 0.5}, {"A": 0.5, "B": 0.5}, {"C": 1}]
    assert _decomp_to_transitions(T, decomp) == [
        [0.0, 0.0, ["A"]],
        [100.0, 200.0, ["A", "B"]],
        [300.0, 300.0, ["C"]],
    ]


def test_decomp_transitions_roundtrip_preserves_product_sets():
    decomp = [{"A": 1}, {"A": 0.5, "B": 0.5}, {"A": 0.5, "B": 0.5}, {"C": 1}]
    trans = _decomp_to_transitions(T, decomp)
    per_T = _transitions_to_per_T(T, trans)
    assert [sorted(d.keys()) for d in per_T] == [["A"], ["A", "B"], ["A", "B"], ["C"]]


def test_decomp_empty():
    assert _decomp_to_transitions(T, []) == []
    assert _transitions_to_per_T(T, []) == []


def test_config_hash_coerces_numpy_types():
    # numpy-typed config values must hash identically to native ones (and not crash json.dumps).
    numpyish = {"mixing": "GGA", "P_O2": np.float64(1.0), "si_reference": None,
                "include_gnome_competitors": np.bool_(False), "T_values": np.linspace(0, 1500, 5)}
    native = {"mixing": "GGA", "P_O2": 1.0, "si_reference": None,
              "include_gnome_competitors": False, "T_values": list(np.linspace(0, 1500, 5))}
    assert _config_hash(numpyish) == _config_hash(native)


# ----------------------------------------------------------------- put / get

def test_put_get_roundtrip(tmp_path):
    rc = _new_store(tmp_path / "r.h5")
    decomp = [{"A": 1}, {"A": 1}, {"B": 1}, {"B": 1}]
    rc.put("m1", _result([-0.1, 0.0, 0.2, 0.5], decomp=decomp, mp_sub=True))

    got = rc.get("m1")
    assert np.allclose(got.E_above_hull, [-0.1, 0.0, 0.2, 0.5])
    assert got.mp_substituted is True
    assert [sorted(d.keys()) for d in got.decomp] == [["A"], ["A"], ["B"], ["B"]]
    rc.close()


def test_has_and_count(tmp_path):
    rc = _new_store(tmp_path / "r.h5")
    assert not rc.has("m1")
    rc.put("m1", _result([0, 0, 0, 0]))
    assert rc.has("m1")
    assert rc.count() == 1
    rc.close()


def test_put_overwrites_existing_row(tmp_path):
    rc = _new_store(tmp_path / "r.h5")
    rc.put("m1", _result([1, 1, 1, 1]))
    rc.put("m1", _result([2, 2, 2, 2]))  # overwrite, not append
    assert rc.count() == 1
    assert np.allclose(rc.get("m1").E_above_hull, [2, 2, 2, 2])
    rc.close()


def test_put_before_bind_raises(tmp_path):
    rc = ResultsCache(tmp_path / "r.h5", force=True)
    with pytest.raises(RuntimeError, match="before bind"):
        rc.put("m1", _result([0, 0, 0, 0]))
    rc.close()


def test_put_wrong_length_raises(tmp_path):
    rc = _new_store(tmp_path / "r.h5")
    with pytest.raises(ValueError, match="T-grid length"):
        rc.put("m1", CurveResult(np.array(T), np.array([0.0, 0.0]), [], False))
    rc.close()


# --------------------------------------------------- config + force-gated bind

def test_config_property(tmp_path):
    rc = ResultsCache(tmp_path / "r.h5", force=True)
    assert rc.config is None                 # not yet initialised
    rc.bind(_config())
    assert rc.config["mixing"] == "GGA" and rc.config["P_O2"] == 1.0
    rc.close()


def test_fresh_store_without_force_raises(tmp_path):
    rc = ResultsCache(tmp_path / "r.h5")     # force=False
    with pytest.raises(ValueError, match="No results store exists yet"):
        rc.bind(_config())
    rc.close()


def test_config_match_attaches(tmp_path):
    rc = _new_store(tmp_path / "r.h5")
    rc.put("m1", _result([0, 0, 0, 0]))
    rc.bind(_config())                        # same config -> attach, keep rows
    assert rc.has("m1")
    rc.close()


def test_config_mismatch_without_force_raises_and_keeps_data(tmp_path):
    path = tmp_path / "r.h5"
    rc = _new_store(path, _config(P_O2=1.0))
    rc.put("m1", _result([0, 0, 0, 0]))
    rc.close()

    rc2 = ResultsCache(path)                  # force=False
    with pytest.raises(ValueError, match="different config"):
        rc2.bind(_config(P_O2=0.21))
    assert rc2.has("m1")                       # NOT wiped
    rc2.close()


def test_config_mismatch_with_force_overwrites(tmp_path):
    path = tmp_path / "r.h5"
    rc = _new_store(path, _config(P_O2=1.0))
    rc.put("m1", _result([0, 0, 0, 0]))
    rc.close()

    rc2 = ResultsCache(path, force=True)
    rc2.bind(_config(P_O2=0.21))              # force -> overwrite
    assert not rc2.has("m1")
    assert rc2.count() == 0
    rc2.close()


def test_bind_force_override_initialises_empty_store(tmp_path):
    # __init__ creates an empty .h5; a force=False cache must still let an explicit force=True bind
    # initialise it (this is what saves the notebook's "file exists but unbound" case).
    rc = ResultsCache(tmp_path / "r.h5")           # force=False; empty file
    assert rc.config is None
    with pytest.raises(ValueError, match="No results store exists yet"):
        rc.bind(_config())                          # default force (False) -> refuses
    rc.bind(_config(), force=True)                  # per-call override -> creates
    assert rc.config is not None
    rc.put("m1", _result([0, 0, 0, 0]))
    assert rc.has("m1")
    rc.close()


# ----------------------------------------------------------------- persistence

def test_reopen_loads_index(tmp_path):
    path = tmp_path / "r.h5"
    rc = _new_store(path)
    rc.put("m1", _result([-0.3, 0.1, 0.4, 0.9]))
    rc.close()

    rc2 = ResultsCache(path)                   # reopen (attach) — index loaded in __init__
    assert rc2.has("m1")
    assert np.allclose(rc2.get("m1").E_above_hull, [-0.3, 0.1, 0.4, 0.9])
    rc2.close()


# ----------------------------------------------------------------- bulk loaders

def test_load_curves_and_dataframe(tmp_path):
    rc = _new_store(tmp_path / "r.h5")
    rc.put("m1", _result([0, 1, 2, 3]))
    rc.put("m2", _result([0, -1, -2, -3], mp_sub=True))

    ids, E, mp_sub = rc.load_curves()
    assert set(ids) == {"m1", "m2"}
    assert E.shape == (2, 4)
    assert mp_sub.dtype == bool

    df = rc.to_dataframe()
    assert set(df["MaterialId"]) == {"m1", "m2"}
    assert list(df.columns) == ["MaterialId", "E_above_hull", "mp_substituted", "decomp"]
    rc.close()


def test_force_recompute_flag_stored(tmp_path):
    rc = ResultsCache(tmp_path / "r.h5", force_recompute=True)
    assert rc.force_recompute is True
    rc.close()


# ----------------------------------------------------------------- list stores

def test_list_result_stores(tmp_path):
    a = _new_store(tmp_path / "ggaair.h5", _config(P_O2=0.21))
    a.put("m1", _result([0, 0, 0, 0]))
    a.close()
    b = _new_store(tmp_path / "ggapure.h5", _config(P_O2=1.0))
    b.close()

    df = list_result_stores(tmp_path).set_index("file")
    assert set(df.index) == {"ggaair.h5", "ggapure.h5"}
    assert df.loc["ggaair.h5", "P_O2"] == 0.21
    assert df.loc["ggaair.h5", "count"] == 1
    assert df.loc["ggapure.h5", "count"] == 0
    assert df.loc["ggapure.h5", "n_T"] == 4


def test_list_result_stores_empty_dir_has_columns(tmp_path):
    df = list_result_stores(tmp_path)
    assert len(df) == 0
    assert list(df.columns) == [
        "file", "mixing", "P_O2", "si_reference", "include_gnome_competitors", "n_T", "count",
    ]
