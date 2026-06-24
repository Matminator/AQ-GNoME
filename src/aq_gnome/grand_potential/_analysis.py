import numpy as np
from pymatgen.core import Element
from pymatgen.analysis.phase_diagram import (
    PhaseDiagram, GrandPotentialPhaseDiagram, GrandPotPDEntry,
)
from aq_gnome.grand_potential._shomate import delta_mu_O


def compute_Si_O_stability_curve(
    gnome_corrected,
    non_target_corrected: list,
    all_entries_corrected: list,
    T_values,
    P_O2: float = 1.0,
    si_reference: str | None = 'SiO2',
    return_decomp: bool = False,
) -> tuple[np.ndarray, list[dict]]:
    """Compute e_above_hull (eV/atom) for a GNoME entry as a function of T.

    O is always an open element with a T-dependent chemical potential from the
    Shomate equation. Si is optionally opened via si_reference. The GPD is built
    from non_target_corrected (GNoME and its MP copies excluded), so a negative
    value means the GNoME candidate is a new stable phase.

    Parameters
    ----------
    gnome_corrected : ComputedStructureEntry
        Corrected GNoME candidate — must NOT appear in non_target_corrected.
    non_target_corrected : list
        Corrected competitor entries (GNoME target and its MP copies excluded).
        Returned as the second element of apply_corrections().
    all_entries_corrected : list
        Full corrected set (MP + GNoME), used for el_refs and SiO2 reference.
    T_values : array-like
        Temperatures in Kelvin. T=0 uses ΔμO = 0 (skips Shomate correction).
    P_O2 : float
        O₂ partial pressure in bar. Default 1.0 = pure O₂.
    si_reference : str or None
        How μSi is set when Si is an open element.
        'SiO2' (default) : μSi set by SiO2 stability (Si in equilibrium with SiO2).
        'Si'              : μSi = DFT reference energy for elemental Si.
        None              : Si is not opened (pure-O grand potential).
    return_decomp : bool
        If True, decomp_list contains one {reduced_formula: fraction} dict per T.
        If False, decomp_list is an empty list.

    Returns
    -------
    E_arr : np.ndarray
        E above MP hull in eV/atom (negative = below hull).
    decomp_list : list of dict
        Decomposition products per temperature. Empty list when return_decomp=False.
    """
    if si_reference not in ('SiO2', 'Si', None):
        raise ValueError(f"si_reference must be 'SiO2', 'Si', or None. Got '{si_reference}'.")

    pd_full = PhaseDiagram(all_entries_corrected)
    mu_O_ref = pd_full.el_refs[Element('O')].energy_per_atom

    # Pre-compute Si reference energy once (before the T loop).
    if si_reference == 'SiO2':
        sio2_candidates = [
            e for e in all_entries_corrected
            if e.composition.reduced_formula == 'SiO2'
        ]
        if not sio2_candidates:
            raise ValueError(
                "No SiO2 entry found in the corrected entries. "
                "Cannot apply the SiO2 equilibrium constraint for μSi. "
                "Use si_reference='Si' instead."
            )
        sio2_ref = min(sio2_candidates, key=lambda e: e.energy_per_atom)
        E_SiO2_per_fu = sio2_ref.energy_per_atom * 3  # 3 atoms per SiO2 formula unit
    elif si_reference == 'Si':
        mu_Si_elem = pd_full.el_refs[Element('Si')].energy_per_atom

    Es = []
    decomp_list = []

    for T in T_values:
        dmu = 0.0 if T == 0 else delta_mu_O(T, P_O2=P_O2)
        mu_O = mu_O_ref + dmu

        ops = {Element('O'): mu_O}
        if si_reference == 'SiO2':
            # μSi set by SiO2 stability: Si + 2O ⇌ SiO2 → μSi = G(SiO2) − 2·μO
            ops[Element('Si')] = E_SiO2_per_fu - 2 * mu_O
        elif si_reference == 'Si':
            ops[Element('Si')] = mu_Si_elem

        gpd = GrandPotentialPhaseDiagram(non_target_corrected, ops)
        gpe = GrandPotPDEntry(gnome_corrected, ops)
        decomp, e = gpd.get_decomp_and_e_above_hull(gpe, allow_negative=True)
        Es.append(e)

        if return_decomp:
            decomp_list.append({
                entry.original_entry.composition.reduced_formula: round(amt, 4)
                for entry, amt in decomp.items()
            })

    return np.array(Es), decomp_list
