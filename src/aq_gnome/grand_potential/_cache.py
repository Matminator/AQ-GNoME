"""Local, lazily-grown cache of raw Materials Project entries.

The grand-potential pipeline needs the MP entries of a chemical system to build the
competing-phase hull. Fetching them live (one ``get_entries_in_chemsys`` call per target)
is redundant and slow at scale. ``MPCache`` downloads each chemical (sub)system once into a
local SQLite store of **raw** ``ComputedStructureEntry`` objects and serves them from disk
thereafter.

Design (see the Step-1 plan):

* Stored entries are **raw** (``compatible_only=False``) and cover all thermo types
  (GGA/GGA+U *and* r2SCAN), so one cache serves both the GGA and GGA+r2SCAN mixing modes.
  Corrections are applied later, at analysis time.
* Downloads are **lazy** and **batched**: only chemsys not already in ``fetched_systems`` are
  requested, deduplicated across the whole batch, via ``MPRester.get_entries`` on the missing
  chemsys list (chunked to stay under the client URI limit). No redundant network.
* Coverage is tracked by **exact chemsys strings**. A target system is covered iff every one
  of its subsystem chemsys has been fetched.
* ``{O, Si}`` are always added to a target's elements so the cache serves Si-open and
  Si-closed runs alike.
"""

from __future__ import annotations

import itertools
import pickle
import sqlite3
import warnings
from pathlib import Path

import pandas as pd
from mp_api.client import MPRester

# Thermo types requested so that raw GGA/GGA+U *and* r2SCAN entries land in the cache.
# Passing additional_criteria also suppresses the client's default-thermotype warning.
_THERMO_TYPES = ["GGA_GGA+U", "GGA_GGA+U_R2SCAN", "R2SCAN"]

_DEFAULT_OPEN_ELEMENTS = ("O", "Si")


def _subsystems(elements) -> list[str]:
    """All non-empty subsystem chemsys strings of ``elements`` (sorted, dash-joined).

    Mirrors exactly what ``MPRester.get_entries_in_chemsys`` enumerates internally, so the
    strings stored in ``fetched_systems`` line up with what was actually queried, and with
    ``Composition.chemical_system`` (both sort element symbols alphabetically).

    Example: ``_subsystems(["Co", "O"]) -> ["Co", "O", "Co-O"]``.
    """
    els = sorted(set(elements))
    return [
        "-".join(combo)
        for i in range(1, len(els) + 1)
        for combo in itertools.combinations(els, i)
    ]


def _element_sets(targets) -> list[set[str]]:
    """Normalise ``targets`` into a list of element sets.

    Accepts a ``pandas.DataFrame`` (uses its ``'Elements'`` column, already lists per
    ``Data_Handler._load_df``) or any iterable of element collections.
    """
    if isinstance(targets, pd.DataFrame):
        return [set(els) for els in targets["Elements"]]
    return [set(t) for t in targets]


def _chunks(seq, size):
    """Yield successive ``size``-length chunks of ``seq``."""
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


class MPCache:
    """SQLite-backed cache of raw Materials Project entries.

    This is the self-contained Materials Project data source: configure it once (path, API key,
    refresh policy) outside any other class, then hand it to consumers such as
    ``GrandPotentialAnalyzer``.

    Parameters
    ----------
    path : str | Path | None
        Location of the cache DB. ``None`` (default) resolves to ``<repo_root>/data/mp_cache.db``.
        Pass an explicit path for tests / scratch DBs, or to place it next to a notebook.
    make_new : bool
        If False and the DB file does not yet exist, raise ``FileNotFoundError`` instead of
        creating it.
    api_key : str | None
        Materials Project API key used for downloads. Stored on the cache so callers (e.g. the
        analyzer) don't need to pass it around; individual ``ensure_cached`` calls may still
        override it.
    force_refresh : bool
        If True, every ``ensure_cached`` call **re-downloads and overwrites** the chemical systems
        it is asked for, even if they are already cached — use this to update stale data. Systems
        that are *not* requested in a given call are left untouched. Default False (normal lazy
        behaviour: cached systems are served from disk with no network).
    """

    def __init__(
        self,
        path: str | Path | None = None,
        make_new: bool = True,
        api_key: str | None = None,
        force_refresh: bool = False,
    ):
        if path is None:
            # _cache.py -> grand_potential -> aq_gnome -> src -> <repo_root>
            repo_root = Path(__file__).resolve().parents[3]
            path = repo_root / "data" / "mp_cache.db"
        self.path = Path(path)
        self.api_key = api_key
        self.force_refresh = force_refresh

        if not self.path.exists() and not make_new:
            raise FileNotFoundError(f"MP cache database not found at {self.path}")

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.cursor = self.conn.cursor()
        self._create_tables()

    def _create_tables(self):
        self.cursor.executescript(
            """
            CREATE TABLE IF NOT EXISTS entries (
                id        TEXT NOT NULL,
                run_type  TEXT NOT NULL,
                chemsys   TEXT NOT NULL,
                entry     BLOB NOT NULL,
                PRIMARY KEY (id, run_type)
            );
            CREATE INDEX IF NOT EXISTS idx_entries_chemsys ON entries(chemsys);

            CREATE TABLE IF NOT EXISTS fetched_systems (
                chemsys TEXT PRIMARY KEY
            );
            """
        )
        self.conn.commit()

    # ------------------------------------------------------------------ writes

    def ensure_cached(
        self,
        targets,
        api_key: str | None = None,
        open_elements=_DEFAULT_OPEN_ELEMENTS,
        max_system_elements: int = 10,
        on_oversize: str = "warn",
        chunk_size: int = 400,
        verbose: bool = True,
    ) -> dict:
        """Download any Materials Project data missing for ``targets`` and store it locally.

        For every target the chemical system is ``set(target elements) | set(open_elements)``.
        Each system is expanded into its subsystem chemsys (all element sub-combinations, e.g.
        ``Co-Br-O-Si`` -> ``O``, ``Co-O``, ``Br-Co-O``, ...). The subsystems not yet present in
        ``fetched_systems`` are pooled across the whole batch, de-duplicated, and downloaded in one
        batched, chunked ``MPRester.get_entries`` pass. A system whose subsystems are all already
        cached triggers no network call at all.

        This is idempotent and resumable: calling it again with the same (or overlapping) targets
        only downloads what is genuinely new.

        Parameters
        ----------
        targets : pandas.DataFrame | iterable of iterable of str
            The materials whose competing-phase data is needed. Either a DataFrame with an
            ``'Elements'`` column (each cell a list of element symbols, as produced by
            ``Data_Handler``), or any iterable of element collections, e.g.
            ``[['Co', 'Br'], {'Co', 'Mo', 'Zn'}]``. Only the element *sets* are used; duplicates
            and ordering are ignored.
        api_key : str | None
            Materials Project API key. Defaults to the cache's stored ``api_key`` when None. Used
            only if something actually needs downloading — a fully-cached batch (with
            ``force_refresh=False``) makes no network call, so the key is not consulted then.
        open_elements : tuple of str, default ('O', 'Si')
            Elements added to every target's element set before forming its chemical system. These
            are the grand-potential open reservoirs; baking them into every system means one cache
            serves both Si-open and Si-closed analyses without re-downloading.
        max_system_elements : int, default 10
            Upper bound on the number of elements in a system (after adding ``open_elements``).
            A system with ``n`` elements expands to ``2**n - 1`` subsystem queries, so very large
            systems are guarded against rather than silently exploding the download.
        on_oversize : {'warn', 'raise'}, default 'warn'
            What to do when a system exceeds ``max_system_elements``. ``'warn'`` emits a
            ``warnings.warn`` naming the system and records it in ``skipped_too_large`` (the batch
            continues); ``'raise'`` raises ``ValueError`` immediately. Either way the system is
            never dropped silently.
        chunk_size : int, default 400
            How many chemsys strings to request per ``get_entries`` call. Kept below the client's
            URI-length limit (~511 chemsys); only affects how the download is batched, not the
            result.
        verbose : bool, default True
            If True, show a tqdm progress bar over the download chunks and print a one-line summary.

        Returns
        -------
        dict
            Summary of the call with keys:

            ``n_systems`` : int
                Number of unique chemical systems derived from ``targets``.
            ``n_skipped_covered`` : int
                How many of those systems were already fully cached (no download needed).
            ``n_skipped_too_large`` : int
                How many systems exceeded ``max_system_elements`` and were skipped.
            ``skipped_too_large`` : list of list of str
                The oversized systems, each as a sorted list of element symbols (so the caller —
                e.g. the Step-6 driver — can map them back to the affected materials).
            ``n_chemsys_fetched`` : int
                Number of chemsys strings actually downloaded in this call (0 if nothing was
                missing).
            ``entries_written`` : int
                Number of MP entry rows written to the database in this call.
            ``total_entries`` : int
                Total number of entries in the cache after this call.

        Raises
        ------
        ValueError
            If ``on_oversize`` is not ``'warn'`` or ``'raise'``, or if ``on_oversize='raise'`` and a
            target system exceeds ``max_system_elements``.

        Examples
        --------
        >>> cache = MPCache()
        >>> cache.ensure_cached([['Co', 'Br'], ['Co', 'Mo', 'Zn']], api_key)  # first run downloads
        >>> cache.ensure_cached([['Co', 'Br']], api_key)['n_chemsys_fetched']  # already cached
        0
        """
        if on_oversize not in ("warn", "raise"):
            raise ValueError(f"on_oversize must be 'warn' or 'raise'. Got {on_oversize!r}.")

        key = api_key if api_key is not None else self.api_key
        open_set = set(open_elements)
        systems = {frozenset(elems | open_set) for elems in _element_sets(targets)}

        fetched = self._load_fetched_systems()

        missing: set[str] = set()
        skipped_covered = 0
        skipped_too_large: list[list[str]] = []

        for system in systems:
            if len(system) > max_system_elements:
                msg = (
                    f"Chemical system {sorted(system)} has {len(system)} elements "
                    f"(> max_system_elements={max_system_elements}) and cannot be cached/analysed."
                )
                if on_oversize == "raise":
                    raise ValueError(msg)
                warnings.warn(msg, stacklevel=2)
                skipped_too_large.append(sorted(system))
                continue

            subs = _subsystems(system)
            # force_refresh re-fetches every requested system; otherwise only the not-yet-cached
            # subsystems are downloaded.
            miss = subs if self.force_refresh else [s for s in subs if s not in fetched]
            if not miss:
                skipped_covered += 1
            else:
                missing.update(miss)

        entries_written = 0
        n_chemsys_fetched = 0
        if missing:
            entries_written, n_chemsys_fetched = self._download(
                sorted(missing), key, chunk_size=chunk_size, verbose=verbose,
            )

        summary = {
            "n_systems": len(systems),
            "n_skipped_covered": skipped_covered,
            "n_skipped_too_large": len(skipped_too_large),
            "skipped_too_large": skipped_too_large,
            "n_chemsys_fetched": n_chemsys_fetched,
            "entries_written": entries_written,
            "total_entries": self.count_entries(),
        }
        if verbose:
            print(
                f"MPCache: {summary['n_systems']} systems "
                f"({skipped_covered} already covered, "
                f"{summary['n_skipped_too_large']} too large) -> "
                f"fetched {n_chemsys_fetched} chemsys, wrote {entries_written} entries "
                f"({summary['total_entries']} total in cache)."
            )
        return summary

    def _download(self, chemsys_list, api_key, chunk_size, verbose) -> tuple[int, int]:
        """Fetch ``chemsys_list`` from MP in chunks and store the entries.

        Single-element chemsys (elemental references) must travel in a chunk that also contains a
        dashed chemsys: ``get_entries`` routes a dash-free list to a *formula* query, which differs
        from a chemsys query for diatomics (``formula "O"`` != ``chemsys "O"`` = O2). We pack all
        singles into the first chunk (they are few — at most one per element — and every system adds
        O+Si, so a dashed chemsys always exists alongside them), keeping every chunk routed as a
        chemsys query.

        Returns ``(entries_written, n_chemsys_fetched)``.
        """
        entries_written = 0
        dashed = [c for c in chemsys_list if "-" in c]
        singles = [c for c in chemsys_list if "-" not in c]
        head = max(0, chunk_size - len(singles))
        chunks = [singles + dashed[:head]] + [list(ch) for ch in _chunks(dashed[head:], chunk_size)]
        chunks = [c for c in chunks if c]
        with MPRester(api_key=api_key) as mpr:
            iterator = chunks
            if verbose:
                try:
                    from tqdm import tqdm
                    iterator = tqdm(chunks, desc="MPCache: downloading chemsys chunks")
                except ImportError:
                    pass
            for chunk in iterator:
                entries = mpr.get_entries(
                    chunk,
                    compatible_only=False,
                    additional_criteria={"thermo_types": _THERMO_TYPES},
                )
                entries_written += self._store_entries(entries)
                # Mark every requested chemsys fetched, even those that returned no entries,
                # so we never re-query an empty subsystem.
                self._mark_fetched(chunk)
                self.conn.commit()
        return entries_written, len(chemsys_list)

    def _store_entries(self, entries) -> int:
        rows = [
            (
                str(e.entry_id),  # MP returns an EntryID object, not a plain str
                str(e.parameters.get("run_type") or ""),
                e.composition.chemical_system,
                pickle.dumps(e, protocol=pickle.HIGHEST_PROTOCOL),
            )
            for e in entries
        ]
        self.cursor.executemany(
            "INSERT OR REPLACE INTO entries (id, run_type, chemsys, entry) VALUES (?, ?, ?, ?)",
            rows,
        )
        return len(rows)

    def _mark_fetched(self, chemsys_list):
        self.cursor.executemany(
            "INSERT OR IGNORE INTO fetched_systems (chemsys) VALUES (?)",
            [(c,) for c in chemsys_list],
        )

    # ------------------------------------------------------------------- reads

    def _load_fetched_systems(self) -> set[str]:
        self.cursor.execute("SELECT chemsys FROM fetched_systems")
        return {row[0] for row in self.cursor.fetchall()}

    def get_entries_for_system(self, elements, open_elements=_DEFAULT_OPEN_ELEMENTS) -> list:
        """Return cached raw entries whose chemsys is a subsystem of ``elements ∪ open_elements``.

        Pure local read, no network. This is the element-subset reader the analyzer uses at
        curve time (Step 2).
        """
        subs = _subsystems(set(elements) | set(open_elements))
        if not subs:
            return []
        placeholders = ",".join("?" * len(subs))
        self.cursor.execute(
            f"SELECT entry FROM entries WHERE chemsys IN ({placeholders})", subs
        )
        return [pickle.loads(row[0]) for row in self.cursor.fetchall()]

    def fetch_for_system(
        self, elements, api_key: str | None = None, open_elements=_DEFAULT_OPEN_ELEMENTS, **ensure_kwargs
    ) -> list:
        """Lazy cache-backed fetcher: return a system's raw entries, downloading on miss.

        Conceptual drop-in *data source* for the live ``fetch_mp_entries`` — but it returns
        **raw** entries (no mixing-mode filtering; that is the analyzer's job). A notebook calls
        this per target; the batch driver instead calls :meth:`ensure_cached` once on the whole
        df and then :meth:`get_entries_for_system` per target (no further downloads).
        """
        self.ensure_cached([elements], api_key, open_elements=open_elements, **ensure_kwargs)
        return self.get_entries_for_system(elements, open_elements=open_elements)

    # ------------------------------------------------------------- diagnostics

    def count_entries(self) -> int:
        self.cursor.execute("SELECT COUNT(*) FROM entries")
        return self.cursor.fetchone()[0]

    def fetched_count(self) -> int:
        self.cursor.execute("SELECT COUNT(*) FROM fetched_systems")
        return self.cursor.fetchone()[0]

    def close(self):
        self.conn.close()
