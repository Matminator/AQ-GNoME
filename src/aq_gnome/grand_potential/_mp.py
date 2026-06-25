import warnings
from mp_api.client import MPRester
from pymatgen.entries.compatibility import MaterialsProject2020Compatibility

from aq_gnome.grand_potential._cache import _THERMO_TYPES


_ALLOWED_RUN_TYPES: dict[str, set[str]] = {
    'GGA':        {'GGA', 'GGA+U'},
    'GGA+r2SCAN': {'GGA', 'GGA+U', 'r2SCAN', 'R2SCAN'},
}
_R2SCAN_FAMILY = {'r2SCAN', 'R2SCAN'}


def filter_run_types(entries: list, mixing: str) -> list:
    """Keep only entries whose run_type is allowed for the mixing mode; drop the rest.

    The cache stores raw entries of *all* run types (GGA/GGA+U and r2SCAN). At analysis time we
    keep what each mode needs: ``'GGA'`` keeps GGA/GGA+U (r2SCAN is disregarded); ``'GGA+r2SCAN'``
    keeps all three. This replaces an older validate-and-raise check, which broke once the MP
    default thermo type began returning r2SCAN entries even for a GGA request.
    """
    allowed = _ALLOWED_RUN_TYPES[mixing]
    return [e for e in entries if e.parameters.get('run_type') in allowed]


def _make_compat(mixing: str):
    """Return the appropriate compatibility / mixing-scheme object.

    Raises ValueError on any unrecognised mixing mode rather than silently falling through to the
    mixing scheme.
    """
    if mixing not in _ALLOWED_RUN_TYPES:
        raise ValueError(f"mixing must be one of {list(_ALLOWED_RUN_TYPES)}. Got '{mixing}'.")
    mp2020 = MaterialsProject2020Compatibility(check_potcar=False)
    if mixing == 'GGA':
        return mp2020
    # Only 'GGA+r2SCAN' remains after the guard above.
    from pymatgen.entries.mixing_scheme import MaterialsProjectDFTMixingScheme
    return MaterialsProjectDFTMixingScheme(compat_1=mp2020, check_potcar=False)


def build_chemical_system_set(entry, additional_elements: tuple[str, ...] = ()) -> set[str]:
    """Return the chemical system set for a GNoME ComputedStructureEntry.

    Adds additional_elements so the MP fetch covers all relevant competing phases.
    """
    elements = {site.specie.symbol for site in entry.structure}
    elements.update(additional_elements)
    return elements


def fetch_mp_entries(chem_sys: set[str], api_key: str) -> list:
    """Fetch RAW MP entries (all run types incl. r2SCAN) for a chemical system, live.

    Low-level convenience that mirrors what ``MPCache`` stores for one system: uncorrected entries
    (``compatible_only=False``) across the GGA/GGA+U and r2SCAN thermo types. Filter by mixing mode
    with :func:`filter_run_types`. Most callers should go through ``MPCache`` instead, which caches
    the result so repeat runs make no network call.

    Parameters
    ----------
    chem_sys : set of str
        Element symbols forming the chemical system.
    api_key : str
        Materials Project API key.
    """
    with MPRester(api_key=api_key) as mpr:
        return mpr.get_entries_in_chemsys(
            list(chem_sys),
            compatible_only=False,
            additional_criteria={"thermo_types": _THERMO_TYPES},
        )


def _structure_matches(gnome_entry, mp_entries: list, structure_matcher) -> list:
    """MP entries that structurally match ``gnome_entry``.

    ``StructureMatcher.fit`` already short-circuits on a composition-hash mismatch before any
    lattice matching, so no manual composition pre-filter is needed here.
    """
    return [m for m in mp_entries if structure_matcher.fit(gnome_entry.structure, m.structure)]


def apply_corrections(
    mp_entries: list,
    target_entries: list,
    competitor_entries: list,
    compat,
    mixing: str,
    structure_matcher=None,
) -> tuple[list, list, object, bool]:
    """Correct MP + GNoME entries and return the hull pieces for one target.

    Parameters
    ----------
    mp_entries : list
        Raw MP entries for the chemical system (all run types).
    target_entries : list
        The target GNoME entries: its GGA entry, plus its r2SCAN entry in ``'GGA+r2SCAN'`` mode when
        the material has one. Feeding both (same structure) lets the mixing scheme anchor an
        otherwise-discarded novel r2SCAN structure.
    competitor_entries : list
        Other in-system GNoME entries to place on the hull (empty unless
        ``include_gnome_competitors``).
    compat : MaterialsProject2020Compatibility | MaterialsProjectDFTMixingScheme
    mixing : str
        ``'GGA'`` or ``'GGA+r2SCAN'``; disallowed run types are dropped from the pool.
    structure_matcher : StructureMatcher or None
        When given, GNoME entries that structurally match an MP entry (same composition) are
        replaced by the MP version: the **target** match sets ``mp_substituted=True`` (the MP twin
        becomes the target); **competitor** matches are dropped in favour of MP.

    Returns
    -------
    all_corrected : list
        Full corrected set (used for el_refs / SiO2 reference).
    non_target_corrected : list
        Corrected competitors (target excluded) — the GrandPotentialPhaseDiagram pool.
    target_corrected : object
        The corrected target entry (prefers the r2SCAN version when both survive).
    mp_substituted : bool
        True if the target's structure already existed in MP and the MP entry was used.

    Raises
    ------
    ValueError
        If the target does not survive the compatibility filter, or >2 MP matches are found for it.
    """
    import copy

    target_label = str(target_entries[0].entry_id)
    # Filter MP entries to the mode's run-types up front, BEFORE structure-matching. Otherwise a
    # target (or competitor) whose structure exists in MP only as an r2SCAN calculation gets
    # substituted by that r2SCAN entry and then deleted by the end-of-function run-type filter in
    # 'GGA' mode — silently losing the phase from its own analysis. Filtering first means 'GGA' mode
    # only ever matches GGA/GGA+U twins; 'GGA+r2SCAN' keeps everything, so this is a no-op there.
    mp_pool = filter_run_types(list(mp_entries), mixing)
    pool_extra: list = []
    target_ids: set[str] = set()
    mp_substituted = False

    # --- Target: use the MP twin(s) if the structure already exists in MP, else add the GNoME
    #     entries (GGA [+ r2SCAN]) so the mixing scheme can self-anchor a novel structure. ---
    target_matches = (
        _structure_matches(target_entries[0], mp_pool, structure_matcher)
        if structure_matcher is not None else []
    )
    if len(target_matches) > 2:
        raise ValueError(
            f"Found {len(target_matches)} MP structure matches for target '{target_label}'. "
            "At most 2 (one GGA/GGA+U and one r2SCAN) are expected."
        )
    if target_matches:
        mp_substituted = True
        matched_ids = {m.entry_id for m in target_matches}
        mp_pool = [e for e in mp_pool if e.entry_id not in matched_ids]
        for m in target_matches:
            sub = copy.copy(m)
            rt = m.parameters.get('run_type')
            suffix = ('_R2SCAN' if rt in _R2SCAN_FAMILY else '_GGA') if len(target_matches) > 1 else ''
            sub.entry_id = f"MP_GNOME{suffix}_{target_label}"
            pool_extra.append(sub)
            target_ids.add(sub.entry_id)
        # If every MP twin is r2SCAN (no GGA/GGA+U among them), the mixing scheme has no GGA entry to
        # anchor that structure and would discard the lone r2SCAN target. Feed the GNoME GGA entry
        # (same structure) as the anchor; the scheme then collapses both to one r2SCAN representative.
        # In 'GGA' mode this never triggers — the run-type pre-filter leaves only GGA/GGA+U twins.
        if all(m.parameters.get('run_type') in _R2SCAN_FAMILY for m in target_matches):
            pool_extra.append(target_entries[0])
            target_ids.add(str(target_entries[0].entry_id))
    else:
        for e in target_entries:
            pool_extra.append(e)
            target_ids.add(str(e.entry_id))

    # --- Competitors: drop any GNoME competitor whose structure already exists in MP. ---
    # TODO: cache per-system structure-match results to avoid recomputing for every target.
    for comp in competitor_entries:
        if structure_matcher is not None and _structure_matches(comp, mp_pool, structure_matcher):
            continue
        pool_extra.append(comp)

    pool = filter_run_types(mp_pool + pool_extra, mixing)

    with warnings.catch_warnings():
        # The GGA+r2SCAN mixing scheme emits a benign "Discarding ... not found in the mixing state
        # data" UserWarning for every redundant polymorph it dedups (dozens per material, mostly the
        # O2/SiO2 references). pymatgen attributes these to this call site, so a module="pymatgen"
        # filter never matches — suppress UserWarning outright for the duration of process_entries.
        warnings.filterwarnings("ignore", category=UserWarning)
        all_corrected = compat.process_entries(pool)

    target_candidates = [e for e in all_corrected if str(e.entry_id) in target_ids]
    if not target_candidates:
        raise ValueError(
            f"GNoME target '{target_label}' did not survive the compatibility filter. "
            "Check that the entry has the expected run_type and POTCAR parameters."
        )
    if len(target_candidates) > 1:
        raise ValueError(
            f"Expected exactly one surviving target entry for '{target_label}', but got "
            f"{len(target_candidates)}: {[e.parameters.get('run_type') for e in target_candidates]}. "
            "The correction/mixing scheme should collapse a structure to a single representative."
        )
    # Exactly one target entry survives — the r2SCAN entry when the target was computed in both
    # functionals (verified), the GGA entry otherwise.
    target_corrected = target_candidates[0]
    # NOTE NOTE NOTE... I am sort of confused by the above lines, I think that one only ever will survive due to the Mixing method. Please check if this is correct.

    survivor_ids = {e.entry_id for e in target_candidates}
    non_target_corrected = [e for e in all_corrected if e.entry_id not in survivor_ids]

    return all_corrected, non_target_corrected, target_corrected, mp_substituted
