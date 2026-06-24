"""Precompute grand-potential curves into a results store (warms the notebook).

Edit the CONFIG block below and run ``python precompute_grand_potential.py``. It screens the
AQ-GNoME database with the same filters as the demo notebook, computes each surviving candidate's
grand-potential curve, and **appends** to a user-named results store (creating it / its directory if
needed, gated by ``FORCE``). The MP download cache is shared across all configs. The notebook,
pointed at the same store + MP cache with a matching config, then loads every curve instantly.

Re-running is safe and resumable: already-stored materials are skipped (cache hits), and a config
that doesn't match an existing store refuses to run unless ``FORCE=True`` (so you never wipe results
by accident). Failures are printed and skipped; the run continues.
"""

import os
from pathlib import Path

import numpy as np

from simple_database import simple_database as sdb
from aq_gnome import Data_Handler, Stable_Entries, Stability_Criteria
from aq_gnome.grand_potential import GrandPotentialAnalyzer, MPCache, ResultsCache

# ============================ CONFIG (printed on run) ============================
# --- AQ-GNoME screening filters (keep entries made exclusively of AVAIL_CHEMICALS,
#     that contain every element in ELEMENTS_MUST_INCLUDE, pass the Pourbaix + HHI cuts) ---
AVAIL_CHEMICALS = (
    ['Ag', 'Bi', 'Br', 'Ce', 'Co', 'Cu', 'Dy', 'Er', 'Fe', 'Ga', 'Ho', 'I', 'In', 'La',
     'Mg', 'Mn', 'Mo', 'Nd', 'Ni', 'Pd', 'Pr', 'Pt', 'Rh', 'Ru', 'Sb', 'Sm', 'Tb', 'Tm',
     'Y', 'Zn', 'Nb', 'Au', 'W', 'Cr', 'Gd', 'Ta', 'Yb', 'Va']
    + ['O', 'N'] + ['Ir', 'Si']
)
ELEMENTS_MUST_INCLUDE = ['Co']
STABILITY_CRITERIA = [Stability_Criteria(pHs=[0], Us=[1.2, 2], decomposition_threshold=0.2)]
HHI_AVG_P_EXCL_OH_MAX = 3500    # keep average_HHI_P_excluding_O_H <  this
HHI_MAX_P_MAX = 5500            # keep max_HHI_P                  <= this

# --- physics (THESE define the store's identity; a different value = a different store) ---
MIXING = 'GGA'                  # 'GGA' or 'GGA+r2SCAN'
P_O2 = 1.0                      # bar  (1.0 = pure O2, 0.21 = air)
SI_REFERENCE = None             # None | 'SiO2' | 'Si'
INCLUDE_GNOME_COMPETITORS = False
T_VALUES = np.linspace(0, 1500, 150)   # set explicitly here (no baked-in default)

# --- paths / store ---
DATA_DIR = None                 # None -> the package's default data/ directory
GNOME_DB = 'processed_GNoME_entries_pp_corr.db'
MP_CACHE_PATH = 'mp_cache.db'   # SHARED across all configs (raw MP data is config-independent)
STORE_DIR = 'results'           # created if missing
STORE_NAME = 'study_GGA.h5'     # user-named results store (one per physics config)
FORCE = False                   # True to create a new store or overwrite a mismatched one
LIMIT = None                    # None = all targets; an int caps how many to compute (handy for tests)
API_KEY = os.environ.get('MP_API_KEY')
# ===============================================================================


def print_config():
    print("=" * 72)
    print("PRECOMPUTE GRAND-POTENTIAL")
    print("=" * 72)
    print(f"  screening : exclusively of {len(set(AVAIL_CHEMICALS))} elements; "
          f"must include {ELEMENTS_MUST_INCLUDE}")
    for sc in STABILITY_CRITERIA:
        print(f"              Pourbaix {sc.col_name} <= {sc.decomposition_threshold} eV/atom")
    print(f"              HHI: avg_P_excl_OH < {HHI_AVG_P_EXCL_OH_MAX}, max_P <= {HHI_MAX_P_MAX}")
    print(f"  physics   : mixing={MIXING}  P_O2={P_O2}  si_reference={SI_REFERENCE}  "
          f"competitors={INCLUDE_GNOME_COMPETITORS}")
    print(f"              T: {T_VALUES[0]:.0f}-{T_VALUES[-1]:.0f} K, {len(T_VALUES)} pts")
    print(f"  store     : {STORE_DIR}/{STORE_NAME}   (force={FORCE}, limit={LIMIT})")
    print(f"  MP cache  : {MP_CACHE_PATH}   GNoME db: {GNOME_DB}")
    print("=" * 72)


def build_targets():
    """Return (full_df, target_ids): the full GNoME df (for the analyzer) and the screened ids."""
    dh = Data_Handler(solid_filter=True, gga_only=False, path_to_data_directory=DATA_DIR)
    full_df = dh.get_df()  # before filtering — needed for in-system competitor enumeration
    dh.remove_entries_not_consisting_exclusively_of_elements(AVAIL_CHEMICALS)
    dh.remove_entries_without_elements(ELEMENTS_MUST_INCLUDE, True)
    df = Stable_Entries(dh, STABILITY_CRITERIA).get_stable_df()
    df = df[df['average_HHI_P_excluding_O_H'] < HHI_AVG_P_EXCL_OH_MAX]
    df = df[df['max_HHI_P'] <= HHI_MAX_P_MAX]
    return full_df, list(df['MaterialId'])


def main():
    print_config()
    if API_KEY is None:
        raise SystemExit("MP_API_KEY is not set in the environment.")

    full_df, target_ids = build_targets()
    if LIMIT is not None:
        target_ids = target_ids[:LIMIT]
    print(f"\n{len(target_ids)} target materials to process.\n")

    db = sdb(GNOME_DB, make_new_db=False)
    Path(STORE_DIR).mkdir(parents=True, exist_ok=True)
    cache = MPCache(path=MP_CACHE_PATH, api_key=API_KEY)
    results = ResultsCache(path=str(Path(STORE_DIR) / STORE_NAME), force=FORCE)

    analyzer = GrandPotentialAnalyzer(
        cache=cache, gnome_db=db, gnome_df=full_df,
        include_gnome_competitors=INCLUDE_GNOME_COMPETITORS,
        mixing=MIXING, P_O2=P_O2, si_reference=SI_REFERENCE, T_values=T_VALUES,
        results_cache=results,
    )
    # Bind up front: fail fast + clearly if the config doesn't match an existing store (FORCE off),
    # or create/overwrite the store (FORCE on).
    results.bind(analyzer.config)

    n_ok = n_fail = 0
    for i, mid in enumerate(target_ids, 1):
        try:
            analyzer.compute_one(mid)
            n_ok += 1
        except Exception as e:
            n_fail += 1
            print(f"  [{i}/{len(target_ids)}] FAILED {mid}: {e!r}")
        if i % 25 == 0:
            print(f"  ... {i}/{len(target_ids)} processed ({n_fail} failed so far)")

    print(f"\nDone. {n_ok} computed/cached, {n_fail} failed. "
          f"Store now holds {results.count()} curves at {STORE_DIR}/{STORE_NAME}.")
    results.close()
    cache.close()


if __name__ == '__main__':
    main()
