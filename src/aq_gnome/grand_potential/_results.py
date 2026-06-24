"""Local store of computed grand-potential curves, mirroring ``MPCache``.

One self-contained ``results.h5`` holds, for a single physics **config**, every target's curve:

    attrs        config + config_hash (mixing, P_O2, si_reference, include_gnome_competitors, T-grid)
    /T_grid          float[N_T]            shared temperature grid
    /material_ids    str[N]               row index (MaterialId)
    /E_above_hull    float[N, N_T]        the curves
    /mp_substituted  int8[N]              was the target replaced by an MP entry?
    /decomp          str[N]               JSON of piecewise decomposition transitions

Design (settled with the user):

* **One store per path.** A store holds results for exactly one config.
* **Force-gated, never silently wiped.** ``bind(config)`` *attaches* when the stored config matches
  (keeping rows); on a mismatch, or on an empty/new file, it **raises** unless the cache was opened
  with ``force=True`` — only then does it truncate and (re)create the store. Keep several configs by
  using several paths; adopt an existing store's config via
  ``GrandPotentialAnalyzer.from_results_cache``.
* **``force_recompute``** (init flag, like ``MPCache.force_refresh``) makes the analyzer recompute
  and overwrite an individual curve even when the config matches.
* Keyed by **MaterialId**; ``put`` overwrites an existing row, else appends (resumable).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from aq_gnome.grand_potential.analyzer import CurveResult


# --------------------------------------------------------------------- config

def _canonical_config(config: dict) -> dict:
    """Canonical, JSON-stable form of a config (T-grid rounded so float noise doesn't fork it)."""
    cfg = dict(config)
    cfg['T_values'] = [round(float(t), 6) for t in config['T_values']]
    return cfg


def _config_hash(config: dict) -> str:
    s = json.dumps(_canonical_config(config), sort_keys=True)
    return hashlib.sha1(s.encode()).hexdigest()[:16]


# --------------------------------------------------------------- decomp <-> H5

def _decomp_to_transitions(T_values, decomp_per_T: list) -> list:
    """Collapse a per-T list of ``{product: fraction}`` dicts into piecewise intervals.

    Returns ``[[T_lo, T_hi, sorted_products], ...]``, one entry per maximal run of an identical
    product set. Only the product *set* is kept (fractions are unused downstream).
    """
    if not decomp_per_T:
        return []
    T = [float(t) for t in T_values]
    sigs = [sorted(d.keys()) for d in decomp_per_T]
    out, start = [], 0
    for i in range(1, len(sigs)):
        if sigs[i] != sigs[start]:
            out.append([T[start], T[i - 1], sigs[start]])
            start = i
    out.append([T[start], T[-1], sigs[start]])
    return out


def _transitions_to_per_T(T_values, transitions: list) -> list:
    """Expand stored transitions back to a per-T list of ``{product: None}`` dicts.

    Fractions are not stored (unused), so they come back as ``None`` — the product keys, which are
    what filters and plots use, are preserved.
    """
    if not transitions:
        return []
    per_T = []
    for t in T_values:
        products = []
        for T_lo, T_hi, prods in transitions:
            if T_lo <= t <= T_hi:
                products = prods
                break
        per_T.append({p: None for p in products})
    return per_T


# ------------------------------------------------------------------- the store

class ResultsCache:
    """SQLite-style results store in a single HDF5 file. See module docstring.

    Parameters
    ----------
    path : str | Path
        Location of the ``.h5`` store (one per physics config).
    force_recompute : bool
        If True, the analyzer recomputes and overwrites individual curves even on a config match.
        Default False.
    force : bool
        Gates destructive store-level operations in :meth:`bind`. When False (default), ``bind``
        only *attaches* to a store whose config matches — it raises rather than create a new store
        or overwrite a mismatched one, so an accidental config change can never wipe accumulated
        results. Set True to deliberately create a new store or overwrite under a different config.
    """

    def __init__(self, path, force_recompute: bool = False, force: bool = False):
        self.path = Path(path)
        self.force_recompute = force_recompute
        self.force = force
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.h5 = h5py.File(self.path, 'a')
        self._index: dict[str, int] = {}
        self._n_T: int | None = None
        if 'material_ids' in self.h5:
            self._load_index()
            self._n_T = self.h5['E_above_hull'].shape[1]

    def _load_index(self):
        ids = [i.decode() if isinstance(i, bytes) else i for i in self.h5['material_ids'][:]]
        self._index = {mid: idx for idx, mid in enumerate(ids)}

    @property
    def config(self) -> dict | None:
        """The physics config the store was created under (``None`` if not yet initialised)."""
        raw = self.h5.attrs.get('config')
        return json.loads(raw) if raw is not None else None

    # ------------------------------------------------------------- config bind

    def bind(self, config: dict):
        """Attach to the store under ``config``, or (with ``force``) create/overwrite it.

        Matching config -> attach (keep rows). Mismatched or empty store -> raise unless ``force``,
        which truncates and recreates the store under ``config``.
        """
        cfg_hash = _config_hash(config)
        if self.h5.attrs.get('config_hash') == cfg_hash and 'material_ids' in self.h5:
            self._n_T = self.h5['E_above_hull'].shape[1]
            return

        if not self.force:
            stored = self.config
            if stored is None:
                raise ValueError(
                    f"No results store exists yet at '{self.path}'. Pass force=True to create one "
                    f"under this config."
                )
            raise ValueError(
                f"Results store '{self.path}' was built under a different config:\n"
                f"  stored:   {stored}\n  requested:{_canonical_config(config)}\n"
                f"Use a different path for a new config, or pass force=True to overwrite (wipes the "
                f"{len(self._index)} stored results)."
            )

        # force=True → start fresh under the new config (create or overwrite).
        self.h5.close()
        self.h5 = h5py.File(self.path, 'w')
        T_grid = np.asarray(config['T_values'], dtype='f8')
        self._n_T = len(T_grid)
        sdt = h5py.string_dtype(encoding='utf-8')
        self.h5.create_dataset('T_grid', data=T_grid)
        self.h5.create_dataset('material_ids', shape=(0,), maxshape=(None,), dtype=sdt)
        self.h5.create_dataset('E_above_hull', shape=(0, self._n_T),
                               maxshape=(None, self._n_T), dtype='f8')
        self.h5.create_dataset('mp_substituted', shape=(0,), maxshape=(None,), dtype='i1')
        self.h5.create_dataset('decomp', shape=(0,), maxshape=(None,), dtype=sdt)
        self.h5.attrs['config_hash'] = cfg_hash
        self.h5.attrs['config'] = json.dumps(_canonical_config(config))
        self.h5.flush()
        self._index = {}

    # --------------------------------------------------------------- read/write

    def has(self, material_id) -> bool:
        return material_id in self._index

    def get(self, material_id) -> CurveResult:
        idx = self._index[material_id]
        E = self.h5['E_above_hull'][idx]
        mp_sub = bool(self.h5['mp_substituted'][idx])
        raw = self.h5['decomp'][idx]
        raw = raw.decode() if isinstance(raw, bytes) else raw
        transitions = json.loads(raw) if raw else []
        T_grid = self.h5['T_grid'][:]
        decomp = _transitions_to_per_T(T_grid, transitions)
        return CurveResult(T_grid, E, decomp, mp_sub)

    def put(self, material_id, result: CurveResult):
        if self._n_T is None:
            raise RuntimeError("ResultsCache.put called before bind().")
        E = np.asarray(result.E_above_hull, dtype='f8')
        if len(E) != self._n_T:
            raise ValueError(f"E length {len(E)} != T-grid length {self._n_T}.")
        decomp_json = json.dumps(_decomp_to_transitions(result.T_values, result.decomp))

        idx = self._index.get(material_id)
        if idx is None:  # append
            idx = len(self._index)
            for name in ('material_ids', 'E_above_hull', 'mp_substituted', 'decomp'):
                self.h5[name].resize(idx + 1, axis=0)
            self._index[material_id] = idx
            self.h5['material_ids'][idx] = material_id
        self.h5['E_above_hull'][idx] = E
        self.h5['mp_substituted'][idx] = int(result.mp_substituted)
        self.h5['decomp'][idx] = decomp_json
        self.h5.flush()

    # ------------------------------------------------------------- bulk loaders

    @property
    def T_grid(self):
        return self.h5['T_grid'][:] if 'T_grid' in self.h5 else None

    def load_curves(self):
        """Return ``(material_ids, E[N, N_T], mp_substituted)`` for vectorised post-hoc sorting."""
        ids = np.array([i.decode() if isinstance(i, bytes) else i
                        for i in self.h5['material_ids'][:]])
        return ids, self.h5['E_above_hull'][:], self.h5['mp_substituted'][:].astype(bool)

    def load_decomp(self) -> dict:
        """``{MaterialId: [[T_lo, T_hi, products], ...]}`` — piecewise decomposition transitions."""
        out = {}
        ids = self.h5['material_ids'][:]
        raws = self.h5['decomp'][:]
        for mid, raw in zip(ids, raws):
            mid = mid.decode() if isinstance(mid, bytes) else mid
            raw = raw.decode() if isinstance(raw, bytes) else raw
            out[mid] = json.loads(raw) if raw else []
        return out

    def to_dataframe(self) -> pd.DataFrame:
        """Metadata table (join to the GNoME df by MaterialId for formula/HHI/Pourbaix)."""
        ids, E, mp_sub = self.load_curves()
        decomp = self.load_decomp()
        return pd.DataFrame({
            'MaterialId': ids,
            'E_above_hull': list(E),
            'mp_substituted': mp_sub,
            'decomp': [decomp[m] for m in ids],
        })

    def count(self) -> int:
        return len(self._index)

    def close(self):
        self.h5.close()


def list_result_stores(directory) -> pd.DataFrame:
    """List the result stores in ``directory`` with each one's physics config and row count.

    Powers the notebook's "which stores do I have, and what physics is in them?" cell. Returns a
    DataFrame with columns: ``file``, ``mixing``, ``P_O2``, ``si_reference``,
    ``include_gnome_competitors``, ``n_T``, ``count``.
    """
    rows = []
    for h5_path in sorted(Path(directory).glob('*.h5')):
        try:
            with h5py.File(h5_path, 'r') as f:
                cfg_raw = f.attrs.get('config')
                if cfg_raw is None:
                    continue  # a plain .h5, not a results store
                cfg = json.loads(cfg_raw)
                n = f['material_ids'].shape[0] if 'material_ids' in f else 0
                n_T = len(cfg.get('T_values', []))
        except OSError:
            continue  # not a readable HDF5 file
        rows.append({
            'file': h5_path.name,
            'mixing': cfg.get('mixing'),
            'P_O2': cfg.get('P_O2'),
            'si_reference': cfg.get('si_reference'),
            'include_gnome_competitors': cfg.get('include_gnome_competitors'),
            'n_T': n_T,
            'count': n,
        })
    return pd.DataFrame(rows)
