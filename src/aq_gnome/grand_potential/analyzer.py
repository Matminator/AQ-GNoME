from dataclasses import dataclass

import numpy as np

from aq_gnome.grand_potential._mp import (
    _ALLOWED_RUN_TYPES, _make_compat, build_chemical_system_set, apply_corrections,
)
from aq_gnome.grand_potential._analysis import compute_Si_O_stability_curve


@dataclass
class CurveResult:
    """Result of one grand-potential curve.

    Attributes
    ----------
    T_values : np.ndarray        Temperatures in Kelvin.
    E_above_hull : np.ndarray    E above the MP(+GNoME) hull in eV/atom (negative = below hull).
    decomp : list                Per-T decomposition products (empty when return_decomp=False).
    mp_substituted : bool        True if the target structure already existed in MP and the MP
                                 entry was used instead of the GNoME one.
    """
    T_values: np.ndarray
    E_above_hull: np.ndarray
    decomp: list
    mp_substituted: bool


class GrandPotentialAnalyzer:
    """Compute grand potential stability curves for GNoME candidates vs the MP(+GNoME) hull.

    The analyzer is driven by **MaterialId**: it holds the GNoME entry db and the full GNoME df, so
    it resolves the right entries per mixing mode (loading both GGA and r2SCAN for the target in
    ``'GGA+r2SCAN'`` mode) and can enumerate every GNoME phase in a chemical system.

    Parameters
    ----------
    cache : MPCache
        Materials Project data source (holds the API key + refresh policy).
    gnome_db : simple_database
        GNoME entry store, keyed by pbx save id (e.g. ``'GGA_...'`` / ``'r2S_...'``).
    gnome_df : pandas.DataFrame
        The **full** GNoME df (e.g. ``Data_Handler.get_df()``), with columns ``MaterialId``,
        ``gga_only_pbx_save_id``, ``mixed_pbx_save_id``, ``Elements``, ``Reduced Formula``. Used for
        MaterialId→entry resolution and (when enabled) in-system competitor enumeration.
    include_gnome_competitors : bool
        Required. True → add every GNoME phase whose elements ⊆ the chemical system to the hull
        (deduped vs MP). False → compare the target against the MP-only hull.
    mixing : str
        ``'GGA'`` (default) or ``'GGA+r2SCAN'``.
    P_O2 : float
        O₂ partial pressure in bar. Default 1.0 = pure O₂.
    T_values : array-like, optional
        Temperatures in Kelvin. Default: [0 K] + linspace(300, 6000, 20).
    si_reference : str or None
        How μSi is set: ``'SiO2'`` (default), ``'Si'``, or ``None`` (Si not opened).
    structure_matcher : StructureMatcher or None
        Used to dedup GNoME entries against MP (use MP when the structure already exists there).
        When None (default) a plain ``StructureMatcher()`` is created — dedup is on by default.
    results_cache : ResultsCache or None
        Optional store of computed curves. When attached, ``compute_one`` returns a stored curve
        instead of recomputing (unless the cache's ``force_recompute`` is set) and stores fresh ones.
    """

    def __init__(
        self,
        cache,
        gnome_db,
        gnome_df,
        include_gnome_competitors: bool,
        mixing: str = 'GGA',
        P_O2: float = 1.0,
        T_values=None,
        si_reference: str | None = 'SiO2',
        structure_matcher=None,
        results_cache=None,
    ):
        if mixing not in _ALLOWED_RUN_TYPES:
            raise ValueError(f"mixing must be one of {list(_ALLOWED_RUN_TYPES)}. Got '{mixing}'.")
        self.cache = cache
        self.gnome_db = gnome_db
        # MaterialId is both index (fast row lookup) and column (kept for enumeration).
        self._df = gnome_df.set_index('MaterialId', drop=False)
        self.include_gnome_competitors = include_gnome_competitors
        self.mixing = mixing
        self.P_O2 = P_O2
        self.si_reference = si_reference
        if structure_matcher is None:
            from pymatgen.analysis.structure_matcher import StructureMatcher
            structure_matcher = StructureMatcher()
        self.structure_matcher = structure_matcher
        self.T_values = (
            [0] + list(np.linspace(300, 6000, 20))
            if T_values is None
            else list(T_values)
        )
        self.compat = _make_compat(mixing)

        self.results_cache = results_cache
        self._results_bound = False
        # The physics config that makes a stored curve valid — see ResultsCache.bind.
        self._config = {
            'mixing': self.mixing,
            'P_O2': self.P_O2,
            'si_reference': self.si_reference,
            'include_gnome_competitors': self.include_gnome_competitors,
            'T_values': list(self.T_values),
        }

    @classmethod
    def from_results_cache(cls, results_cache, cache, gnome_db, gnome_df, structure_matcher=None):
        """Build an analyzer that **adopts an existing store's physics config**, so it only ever
        appends to it (never resets).

        Reads ``mixing / P_O2 / si_reference / include_gnome_competitors / T_values`` from
        ``results_cache`` and configures the analyzer to match. Raises if the store has no config
        yet — initialise it first with the explicit-config constructor and ``ResultsCache(...,
        force=True)``.
        """
        cfg = results_cache.config
        if cfg is None:
            raise ValueError(
                f"Results store '{results_cache.path}' has no config yet. Initialise it first with "
                "GrandPotentialAnalyzer(...) under an explicit config and ResultsCache(..., force=True)."
            )
        return cls(
            cache=cache,
            gnome_db=gnome_db,
            gnome_df=gnome_df,
            include_gnome_competitors=cfg['include_gnome_competitors'],
            mixing=cfg['mixing'],
            P_O2=cfg['P_O2'],
            # Use the store's exact /T_grid (not the 6-dp-rounded config T_values) so curves and
            # decomp transitions added now line up exactly with what's already stored.
            T_values=list(results_cache.T_grid),
            si_reference=cfg['si_reference'],
            structure_matcher=structure_matcher,
            results_cache=results_cache,
        )

    @property
    def config(self) -> dict:
        """The physics config that identifies a results store for this analyzer (the 5 knobs)."""
        return dict(self._config)

    def _resolve_entries(self, row) -> list:
        """Load the GNoME entries for a material's df row.

        Always its GGA entry; plus its r2SCAN entry in ``'GGA+r2SCAN'`` mode when the material has
        one (``mixed_pbx_save_id`` starts ``'r2S_'``). The two share a structure, which lets the
        mixing scheme anchor it.
        """
        gga_id = row['gga_only_pbx_save_id']
        gga_entry = self.gnome_db.get_entry(gga_id)
        if gga_entry is None:
            raise ValueError(f"GNoME entry '{gga_id}' not found in the GNoME db.")
        entries = [gga_entry]
        if self.mixing == 'GGA+r2SCAN':
            mixed_id = row['mixed_pbx_save_id']
            if isinstance(mixed_id, str) and mixed_id.startswith('r2S_'):
                r2s_entry = self.gnome_db.get_entry(mixed_id)
                if r2s_entry is None:
                    raise ValueError(f"GNoME r2SCAN entry '{mixed_id}' not found in the GNoME db.")
                entries.append(r2s_entry)
        return entries

    def _enumerate_competitor_entries(self, chem_sys: set, target_material_id) -> list:
        """GNoME entries for every material whose elements ⊆ chem_sys (excluding the target).

        Empty unless ``include_gnome_competitors``. Each competitor is loaded with the same
        per-mode rule as the target (GGA, plus r2SCAN when available).
        """
        if not self.include_gnome_competitors:
            return []
        in_system = self._df['Elements'].apply(lambda els: set(els) <= chem_sys)
        entries = []
        for material_id, row in self._df[in_system].iterrows():
            if material_id == target_material_id:
                continue
            entries.extend(self._resolve_entries(row))
        # TODO: cache this per-system result keyed by frozenset(chem_sys) to avoid recomputing
        #       it for every target that shares the system.
        return entries

    def compute_one(self, material_id, return_decomp: bool = False) -> CurveResult:
        """Run the full pipeline for one GNoME material (by MaterialId).

        When a ``results_cache`` is attached, a stored curve is returned without recomputing
        (unless the cache's ``force_recompute`` is set); freshly computed curves are stored.
        """
        rc = self.results_cache
        if rc is not None:
            if not self._results_bound:
                rc.bind(self._config)
                self._results_bound = True
            if not rc.force_recompute and rc.has(material_id):
                return rc.get(material_id)

        row = self._df.loc[material_id]
        target_entries = self._resolve_entries(row)

        additional_elements = ('O', 'Si') if self.si_reference is not None else ('O',)
        chem_sys = build_chemical_system_set(target_entries[0], additional_elements=additional_elements)

        self.cache.ensure_cached([chem_sys])
        mp_entries = self.cache.get_entries_for_system(chem_sys, open_elements=())
        competitors = self._enumerate_competitor_entries(chem_sys, material_id)

        all_corrected, non_target_corrected, target_corrected, mp_substituted = apply_corrections(
            mp_entries, target_entries, competitors, self.compat, self.mixing,
            structure_matcher=self.structure_matcher,
        )
        # Collect decomp if the caller asked for it, or if we're going to store it.
        store_decomp = return_decomp or rc is not None
        E_arr, decomp_list = compute_Si_O_stability_curve(
            target_corrected,
            non_target_corrected,
            all_corrected,
            self.T_values,
            P_O2=self.P_O2,
            si_reference=self.si_reference,
            return_decomp=store_decomp,
        )
        result = CurveResult(np.array(self.T_values), E_arr, decomp_list, mp_substituted)
        if rc is not None:
            rc.put(material_id, result)
        return result

    def compute_batch(self, material_ids, return_decomp: bool = False) -> list[dict]:
        """Run compute_one for every MaterialId in ``material_ids``.

        Returns a list of dicts: ``material_id``, ``formula``, ``T_values``, ``E_above_hull``,
        ``decomp``, ``mp_substituted``.
        """
        results = []
        for material_id in material_ids:
            row = self._df.loc[material_id]
            res = self.compute_one(material_id, return_decomp=return_decomp)
            results.append({
                'material_id': material_id,
                'formula': row['Reduced Formula'],
                'T_values': res.T_values,
                'E_above_hull': res.E_above_hull,
                'decomp': res.decomp,
                'mp_substituted': res.mp_substituted,
            })
        return results
