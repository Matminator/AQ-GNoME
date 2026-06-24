import numpy as np

# NIST WebBook Shomate equation coefficients for O₂ (g).
# Source: https://webbook.nist.gov/cgi/cbook.cgi?ID=C7782447&Mask=1#Thermo-Gas
# Chase, M.W. Jr., NIST-JANAF Thermochemical Tables, 4th Ed. (1998); refitted Jan 2009.
#
# H°(T)−H°(298.15K) [kJ/mol] = A*t + B*t²/2 + C*t³/3 + D*t⁴/4 − E/t + F
# S°(T)              [J/mol/K] = A*ln(t) + B*t + C*t²/2 + D*t³/3 − E/(2*t²) + G
# where t = T(K) / 1000.  Valid range: 100–6000 K.
#
# Columns: A, B, C, D, E, F, G
_SHOMATE = np.array([
    # 100–700 K
    [31.32234, -20.23531,  57.86644, -36.50624,  -0.007374,  -8.903471, 246.7945],
    # 700–2000 K
    [30.03235,   8.772972, -3.988133,  0.788313,  -0.741599, -11.32468,  236.1663],
    # 2000–6000 K
    [20.91111,  10.72071,  -2.020498,  0.146449,   9.245722,   5.337651, 237.6185],
])
# H°(0K)−H°(298.15K) = −8.683 kJ/mol
# Source: NIST JANAF Thermochemical Tables, 4th Ed., Table O-029 (T=0K row)
# https://janaf.nist.gov/tables/O-029.html
# Used to re-reference Shomate H from 298.15K to 0K:
#   H°(T)−H°(0K) = Shomate_H(T) + 8.683
_H_0K_CORRECTION = 8.683  # kJ/mol

_EV_PER_KJMOL_PER_O = 1.0 / (96.485 * 2)  # kJ/mol (O₂) → eV/O atom
_k_B = 8.61733326e-5                         # eV/K


def _shomate_raw(T_K: float) -> tuple[float, float]:
    """Shomate H and S for O₂ at T_K ∈ [100, 6000] K. No edge-case handling.

    Returns
    -------
    H_from_0 : float
        H°(T) - H°(0K) in kJ/mol (per O₂ molecule).
    S : float
        S°(T) in J/mol/K (per O₂ molecule).
    """
    t = T_K / 1000.0
    idx = 0 if T_K < 700 else (1 if T_K < 2000 else 2)
    A, B, C, D, E, F, G = _SHOMATE[idx]
    H_from_298 = A*t + B*t**2/2 + C*t**3/3 + D*t**4/4 - E/t + F
    H_from_0   = H_from_298 + _H_0K_CORRECTION
    S          = A*np.log(t) + B*t + C*t**2/2 + D*t**3/3 - E/(2*t**2) + G
    return H_from_0, S


def dH0_O2(T_K: float, eV: bool = True) -> float:
    """H°(T) - H°(0K) for O₂ from NIST WebBook Shomate equations.

    Parameters
    ----------
    T_K : float
        Temperature in Kelvin. Valid range: [0, 6000] K.
        T ∈ (0, 100) K: linear interpolation to the Shomate value at 100 K.
    eV : bool
        If True (default), return eV per O atom.
        If False, return kJ/mol per O₂ molecule (native NIST Shomate units).
    """
    if T_K < 0 or T_K > 6000:
        raise ValueError(f"Temperature must be in [0, 6000] K. Got {T_K} K.")
    if T_K == 0.0:
        return 0.0
    if T_K < 100.0:
        return dH0_O2(100.0, eV=eV) * T_K / 100.0
    H_raw, _ = _shomate_raw(T_K)
    return H_raw * _EV_PER_KJMOL_PER_O if eV else H_raw


def S_O2(T_K: float, eV: bool = True) -> float:
    """S°(T) for O₂ from NIST WebBook Shomate equations.

    Parameters
    ----------
    T_K : float
        Temperature in Kelvin. Valid range: [0, 6000] K.
        T=0K returns 0.0 (third law). T ∈ (0, 100) K: linear interpolation
        to the Shomate value at 100 K.
    eV : bool
        If True (default), return eV/K per O atom.
        If False, return J/mol/K per O₂ molecule (native NIST Shomate units).
    """
    if T_K < 0 or T_K > 6000:
        raise ValueError(f"Temperature must be in [0, 6000] K. Got {T_K} K.")
    if T_K == 0.0:
        return 0.0
    if T_K < 100.0:
        return S_O2(100.0, eV=eV) * T_K / 100.0
    _, S_raw = _shomate_raw(T_K)
    return S_raw * _EV_PER_KJMOL_PER_O / 1000.0 if eV else S_raw


def delta_mu_O(T_K: float, P_O2: float = 1.0) -> float:
    """Thermal + pressure correction to the O chemical potential (eV/atom).

    Computes ΔμO(T, P) relative to the 0K DFT O₂ reference:
        ΔμO = [H°(T)−H°(0K)] − T·S°(T)  +  ½ kB·T·ln(P/P₀)

    H and S from NIST WebBook Shomate equations for O₂ (g).
    At T=0K returns 0.0 by definition (DFT is the 0K reference).

    Source: https://webbook.nist.gov/cgi/cbook.cgi?ID=C7782447&Mask=1#Thermo-Gas

    Parameters
    ----------
    T_K : float
        Temperature in Kelvin. Valid range: [0, 6000] K.
        T ∈ (0, 100) K uses linear interpolation to the Shomate value at 100 K.
    P_O2 : float
        O₂ partial pressure in bar. Default 1.0 = pure O₂.
        0.21 = ambient air (21 vol% O₂ at 1 bar total pressure).
    """
    if T_K < 0 or T_K > 6000:
        raise ValueError(f"Temperature must be in [0, 6000] K. Got {T_K} K.")
    if T_K == 0.0:
        return 0.0
    dmu = dH0_O2(T_K) - T_K * S_O2(T_K) + 0.5 * _k_B * T_K * np.log(P_O2)
    return dmu
