"""Vectorised Pourbaix screening over the precomputed decomposition-energy grids.

Same physics and selection rules as :class:`aq_gnome.stability.Stable_Entries`, but the whole
``[N, 31, 19]`` grid is loaded into memory once (float32, ~1.25 GB for all of GNoME) so that a
stability criterion is evaluated with a single numpy reduction instead of one HDF5 read per
material (~0.3 s for the full database instead of ~6 min). Intended for interactive use (the
Streamlit app in ``app/``); the loop-based ``Stable_Entries`` remains the reference implementation
and ``tests/test_fast_screening.py`` keeps the two in agreement.
"""
import numpy as np
import pandas as pd

from aq_gnome.data_handler import Data_Handler
from aq_gnome.stability import Stability_Criteria

NOT_COMPUTED = 'Not computed'


def load_decomposition_grid(dh: Data_Handler, df: pd.DataFrame | None = None,
                            block: int = 16384) -> np.ndarray:
    """Decomposition-energy grids for every row of ``df`` (default ``dh.get_df()``), as a float32
    array of shape ``[len(df), 31, 19]`` in ``df`` row order.

    Mirrors ``Stable_Entries.get_decom_G``: the mixed-functional grid is used where it exists,
    otherwise the GGA-only grid (always, when ``dh.gga_only``).
    """
    if df is None:
        df = dh.get_df()
    n = len(df)
    grid_shape = dh.gga_results.h5file['decomposition_energies'].shape[1:]
    grid = np.empty((n, *grid_shape), dtype=np.float32)
    _gather(dh.gga_results, df['gga_only_pbx_save_id'].to_numpy(), np.arange(n), grid, block)
    if not dh.gga_only:
        ids = df['mixed_pbx_save_id'].to_numpy()
        has_mixed = np.flatnonzero(ids != NOT_COMPUTED)
        _gather(dh.mixed_results, ids[has_mixed], has_mixed, grid, block)
    return grid


def _gather(h5db, ids, positions, out, block, max_span: int = 65536):
    """``out[positions[k]] = grid of ids[k]``, reading the dataset in sorted contiguous slabs.

    The screening files are chunked one material per chunk, which makes h5py fancy indexing
    ~100x slower than slab reads; ``max_span`` bounds the rows read per slab when ids are sparse.
    """
    h5_idx = np.fromiter((h5db.ids_to_index[i] for i in ids), dtype=np.int64, count=len(ids))
    order = np.argsort(h5_idx, kind='stable')
    h5_idx, positions = h5_idx[order], np.asarray(positions)[order]
    dset = h5db.h5file['decomposition_energies']
    start = 0
    while start < len(h5_idx):
        stop = min(start + block, len(h5_idx),
                   int(np.searchsorted(h5_idx, h5_idx[start] + max_span, side='left')))
        stop = max(stop, start + 1)
        lo, hi = int(h5_idx[start]), int(h5_idx[stop - 1]) + 1
        slab = dset[lo:hi]
        out[positions[start:stop]] = slab[h5_idx[start:stop] - lo]
        start = stop


def element_membership(df: pd.DataFrame) -> pd.DataFrame:
    """Boolean table ``[len(df), n_elements]``, True where the material contains the element."""
    elements = sorted({e for lst in df['Elements'] for e in lst})
    col = {e: i for i, e in enumerate(elements)}
    mat = np.zeros((len(df), len(elements)), dtype=bool)
    for r, lst in enumerate(df['Elements']):
        for e in lst:
            mat[r, col[e]] = True
    return pd.DataFrame(mat, columns=elements, index=df.index)


def element_mask(membership: pd.DataFrame, exclude=(), require=(), require_all: bool = True,
                 allowed_only=None) -> np.ndarray:
    """Row mask with the semantics of the ``Data_Handler.remove_entries_*`` filters.

    exclude      → drop materials containing any of these elements
    require      → keep materials containing all (``require_all``) or any of these elements;
                   requiring an element absent from the data yields an empty selection
    allowed_only → keep materials composed exclusively of these elements (None = no restriction)
    """
    cols = set(membership.columns)
    mask = np.ones(len(membership), dtype=bool)
    ex = [e for e in exclude if e in cols]
    if ex:
        mask &= ~membership[ex].to_numpy().any(axis=1)
    require = list(require)
    if require:
        present = [e for e in require if e in cols]
        if require_all:
            if len(present) < len(require):
                return np.zeros(len(membership), dtype=bool)
            mask &= membership[present].to_numpy().all(axis=1)
        else:
            mask &= membership[present].to_numpy().any(axis=1) if present else False
    if allowed_only is not None:
        not_allowed = [e for e in membership.columns if e not in set(allowed_only)]
        if not_allowed:
            mask &= ~membership[not_allowed].to_numpy().any(axis=1)
    return mask


def region_max(grid: np.ndarray, sc: Stability_Criteria) -> np.ndarray:
    """Vectorised twin of ``Stability_Criteria.max_dG_in_region``: the maximum decomposition
    energy inside the criterion's U×pH region, for every row of ``grid``. Returned as float64 so
    the threshold comparison is done exactly as in the reference (``float(np.max(region))``).
    """
    if isinstance(sc.Us, list) and isinstance(sc.pHs, list):
        region = grid[:, sc.U_lowlimt_index:sc.U_highlimt_index,
                      sc.pH_lowlimt_index:sc.pH_highlimt_index]
        out = region.max(axis=(1, 2))
    elif isinstance(sc.Us, list):
        out = grid[:, sc.U_lowlimt_index:sc.U_highlimt_index, sc.pH_index].max(axis=1)
    elif isinstance(sc.pHs, list):
        out = grid[:, sc.U_index, sc.pH_lowlimt_index:sc.pH_highlimt_index].max(axis=1)
    else:
        out = grid[:, sc.U_index, sc.pH_index]
    return out.astype(np.float64)


def screen(grid: np.ndarray, criteria):
    """Evaluate one or more criteria over the whole grid.

    Returns ``(mask, maxes)``: ``mask[i]`` is True when row ``i`` passes every criterion (max ΔG in
    the region not above the threshold, exactly as ``Stable_Entries.find_stable_entries``), and
    ``maxes[k]`` is the per-row max-ΔG array of criterion ``k``.
    """
    criteria = criteria if isinstance(criteria, list) else [criteria]
    maxes = [region_max(grid, sc) for sc in criteria]
    mask = np.ones(len(grid), dtype=bool)
    for sc, m in zip(criteria, maxes):
        mask &= ~(m > sc.decomposition_threshold)
    return mask, maxes


def stable_dataframe(df: pd.DataFrame, mask: np.ndarray, criteria, maxes,
                     non_negative: bool = True) -> pd.DataFrame:
    """Rows of ``df`` passing the screen with one ``max_dG_…`` column per criterion inserted at the
    front — same layout, clamping and rounding as ``Stable_Entries.get_stable_df``."""
    criteria = criteria if isinstance(criteria, list) else [criteria]
    out = df[mask].copy()
    for k, (sc, m) in enumerate(zip(criteria, maxes)):
        vals = m[mask]
        out.insert(k, sc.col_name, [
            round(max(0.0, float(x)) if non_negative else float(x), 2) for x in vals
        ])
    return out
