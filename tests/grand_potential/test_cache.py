"""Tests for the ``MPCache`` class in ``aq_gnome.grand_potential._cache``.

Cache logic runs with no network. The end-to-end smoke test is gated behind the
``requires_api`` marker and skips automatically when ``MP_API_KEY`` is absent.
"""

import os

import pytest

from aq_gnome.grand_potential._cache import MPCache, _subsystems

MP_API_KEY = os.environ.get("MP_API_KEY")
_R2SCAN_FAMILY = {"r2SCAN", "R2SCAN"}


# ----------------------------------------------------------------- cache logic (no net)

def test_coverage_skips_fully_cached_system(tmp_path):
    cache = MPCache(path=tmp_path / "mp_cache.db")
    # Pretend the whole Co-O-Si system was already downloaded.
    cache._mark_fetched(_subsystems({"Co", "O", "Si"}))
    cache.conn.commit()

    # A target whose system is exactly Co-O-Si needs no fetching.
    summary = cache.ensure_cached([["Co"]], api_key="unused", verbose=False)
    assert summary["n_chemsys_fetched"] == 0
    assert summary["entries_written"] == 0
    assert summary["n_skipped_covered"] == 1
    cache.close()


def test_oversize_system_warns_and_is_surfaced(tmp_path):
    cache = MPCache(path=tmp_path / "mp_cache.db")
    big = ["Li", "Be", "B", "C", "N", "F", "Na", "Mg", "Al", "P", "Cl"]  # +O,Si -> 13 elements
    with pytest.warns(UserWarning, match="cannot be cached"):
        summary = cache.ensure_cached([big], api_key="unused", verbose=False)
    assert summary["n_skipped_too_large"] == 1
    assert sorted(set(big) | {"O", "Si"}) in summary["skipped_too_large"]
    assert summary["n_chemsys_fetched"] == 0
    cache.close()


def test_oversize_system_can_raise(tmp_path):
    cache = MPCache(path=tmp_path / "mp_cache.db")
    big = ["Li", "Be", "B", "C", "N", "F", "Na", "Mg", "Al", "P", "Cl"]
    with pytest.raises(ValueError, match="cannot be cached"):
        cache.ensure_cached([big], api_key="unused", on_oversize="raise", verbose=False)
    cache.close()


def test_invalid_on_oversize_rejected(tmp_path):
    cache = MPCache(path=tmp_path / "mp_cache.db")
    with pytest.raises(ValueError, match="on_oversize"):
        cache.ensure_cached([["Co"]], api_key="unused", on_oversize="explode")
    cache.close()


def test_make_new_false_missing_db_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        MPCache(path=tmp_path / "does_not_exist.db", make_new=False)


def test_force_refresh_redownloads_cached_system(tmp_path, monkeypatch):
    cache = MPCache(path=tmp_path / "mp_cache.db", force_refresh=True)
    cache._mark_fetched(_subsystems({"Co", "O", "Si"}))  # mark as already downloaded
    cache.conn.commit()

    captured = {}

    def fake_download(chemsys_list, api_key, chunk_size, verbose):
        captured["chemsys"] = list(chemsys_list)
        return (0, len(chemsys_list))

    monkeypatch.setattr(cache, "_download", fake_download)
    summary = cache.ensure_cached([["Co"]], api_key="K", verbose=False)

    # Despite being cached, force_refresh re-requests every subsystem of the system.
    assert summary["n_skipped_covered"] == 0
    assert set(captured["chemsys"]) == set(_subsystems({"Co", "O", "Si"}))
    cache.close()


def test_no_force_refresh_skips_cached_system(tmp_path, monkeypatch):
    cache = MPCache(path=tmp_path / "mp_cache.db")  # force_refresh=False (default)
    cache._mark_fetched(_subsystems({"Co", "O", "Si"}))
    cache.conn.commit()

    calls = []
    monkeypatch.setattr(cache, "_download", lambda *a, **k: calls.append(a) or (0, 0))
    cache.ensure_cached([["Co"]], api_key="K", verbose=False)
    assert calls == []  # never hit the network
    cache.close()


def test_api_key_resolution_init_and_override(tmp_path, monkeypatch):
    cache = MPCache(path=tmp_path / "mp_cache.db", api_key="INIT_KEY")
    captured = {}

    def fake_download(chemsys_list, api_key, chunk_size, verbose):
        captured["api_key"] = api_key
        return (0, len(chemsys_list))

    monkeypatch.setattr(cache, "_download", fake_download)
    cache.ensure_cached([["Co", "Br"]], verbose=False)          # no per-call key -> init key
    assert captured["api_key"] == "INIT_KEY"
    cache.ensure_cached([["Ni", "Mo"]], api_key="OVERRIDE", verbose=False)  # override wins
    assert captured["api_key"] == "OVERRIDE"
    cache.close()


def test_download_chunks_never_all_single(tmp_path, monkeypatch):
    """Every chunk sent to get_entries must contain a dashed chemsys, so it routes as a chemsys
    (not formula) query — otherwise diatomic elemental refs (O, N) would be silently lost."""
    import aq_gnome.grand_potential._cache as cache_mod

    sent = []

    class _FakeMPR:
        def __init__(self, api_key=None):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get_entries(self, chunk, **kw):
            sent.append(list(chunk))
            return []

    monkeypatch.setattr(cache_mod, "MPRester", _FakeMPR)
    cache = MPCache(path=tmp_path / "c.db")
    # 8 dashed (an exact multiple of chunk_size) + trailing single-element chemsys.
    chemsys = [f"E{i}-O" for i in range(8)] + ["Co", "N", "O", "Si"]
    cache._download(sorted(chemsys), api_key="k", chunk_size=8, verbose=False)

    assert sent  # something was sent
    assert all(any("-" in c for c in chunk) for chunk in sent)
    cache.close()


# ------------------------------------------------------------------- network smoke test

@pytest.mark.requires_api
@pytest.mark.skipif(not MP_API_KEY, reason="MP_API_KEY not set in environment")
def test_network_smoke(tmp_path):
    cache = MPCache(path=tmp_path / "mp_cache.db")

    summary = cache.ensure_cached([["Co", "Br"], ["Co", "Mo", "Zn"]], MP_API_KEY, verbose=False)
    assert summary["entries_written"] > 0
    assert summary["n_chemsys_fetched"] > 0
    assert summary["skipped_too_large"] == []

    # Raw r2SCAN entries must be captured for the later mixing steps.
    cache.cursor.execute("SELECT DISTINCT run_type FROM entries")
    run_types = {row[0] for row in cache.cursor.fetchall()}
    assert run_types & _R2SCAN_FAMILY, f"no r2SCAN entries cached; got {run_types}"

    # Element-subset read returns only in-system entries, incl. the elemental O reference.
    entries = cache.get_entries_for_system(["Co", "Br"])
    assert entries
    allowed = {"Co", "Br", "O", "Si"}
    assert all({el.symbol for el in e.composition.elements} <= allowed for e in entries)
    assert any(e.composition.chemical_system == "O" for e in entries)

    # Re-running the same targets fetches nothing (lazy coverage; no duplicate rows).
    n_before = cache.count_entries()
    summary2 = cache.ensure_cached([["Co", "Br"], ["Co", "Mo", "Zn"]], MP_API_KEY, verbose=False)
    assert summary2["n_chemsys_fetched"] == 0
    assert summary2["n_skipped_covered"] == 2
    assert cache.count_entries() == n_before
    cache.close()

    # force_refresh re-downloads + overwrites the same system without adding rows.
    refresher = MPCache(path=tmp_path / "mp_cache.db", api_key=MP_API_KEY, force_refresh=True)
    n_rows = refresher.count_entries()
    summary3 = refresher.ensure_cached([["Co", "Br"]], verbose=False)
    assert summary3["n_chemsys_fetched"] > 0      # re-fetched despite being cached
    assert summary3["n_skipped_covered"] == 0
    assert refresher.count_entries() == n_rows    # overwrite, not append
    refresher.close()
