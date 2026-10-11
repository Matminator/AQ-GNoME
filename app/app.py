"""AQ-GNoME — interactive aqueous (Pourbaix) stability screening of GNoME structures.

A thin Streamlit layer over ``aq_gnome``: the same element filters, stability criteria and
post-filters as ``notebooks/data_analysis_v2.ipynb``, evaluated with ``aq_gnome.fast_screening``
so that any query over the full database answers in well under a second. The layout follows the
OER-GNoME database page: dataset choice on top, a clickable periodic table, criteria, optional
post-filters, then the results table with a per-material stability map.

Run locally:   streamlit run app/app.py

Configuration (environment variables):
  AQ_GNOME_DATA_DIR      folder with the DTU Data deposit files (default: <repo>/data)
  AQ_GNOME_MAX_VARIANTS  how many (solid filter, functional) dataset variants to keep in RAM at
                         once, ~1.8 GB each (default 1; switching variant then reloads, ~20 s)
"""
import io
import os
import sys
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import streamlit as st
from ase.io import write as ase_write

from aq_gnome import Data_Handler, Stability_Criteria, get_simplified_df, atoms_from_db
from aq_gnome.fast_screening import (
    load_decomposition_grid, element_membership, element_mask, screen, stable_dataframe,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from periodic_table import periodic_table  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("AQ_GNOME_DATA_DIR", REPO_ROOT / "data"))
MAX_VARIANTS = int(os.environ.get("AQ_GNOME_MAX_VARIANTS", "1"))

FUNCTIONALS = {"GGA/GGA+U + r2SCAN (mixed)": False, "GGA/GGA+U only": True}   # label -> gga_only

# Grids of the precomputed Pourbaix data (see Stability_Criteria)
U_MIN, U_MAX, U_STEP = -2.0, 4.0, 0.2
PH_MIN, PH_MAX, PH_STEP = -2.0, 16.0, 1.0
HHI_MIN, HHI_MAX = 700, 9800
BANDGAP_CAP = 10.0        # slider upper end; at the cap the band-gap filter is off
MAX_CIF_ROWS = 500

LEAD_COLUMNS = ['Reduced Formula', 'Elements', 'NSites', 'Crystal System', 'Dimensionality Cheon',
                'Bandgap', 'Disorder Probability', 'average_HHI_P_excluding_O_H', 'max_HHI_P',
                'MaterialId']
COLUMN_LABELS = {
    'average_HHI_P_excluding_O_H': 'Avg. HHI (production, excl. O/H)',
    'max_HHI_P': 'Max. HHI (production)',
    'Dimensionality Cheon': 'Dimensionality',
    'Disorder Probability': 'Disorder prob.',
}
POST_FILTER_KEYS = ["dims", "n_el", "max_nsites", "bandgap", "max_disorder", "max_hhi_avg",
                    "max_hhi_max"]


# ── Data loading (cached per process, shared by all sessions) ──────────────────

# Page-level CSS: hide Streamlit's running-figure status widget and show a slim progress line
# along the top edge instead while the script is running (the widget only exists in the DOM
# while a run is in progress, which `:has()` turns into a CSS switch).
PAGE_CSS = """
<div class="aq-busy"></div>
<style>
[data-testid="stStatusWidget"] { display: none; }
.aq-busy { position: fixed; top: 0; left: 0; right: 0; height: 3px; z-index: 999999; display: none;
           background: linear-gradient(90deg, transparent 0%, #2a7de1 50%, transparent 100%);
           background-size: 40% 100%; background-repeat: no-repeat;
           animation: aq-slide 1.1s linear infinite; }
body:has([data-testid="stStatusWidget"]) .aq-busy { display: block; }
@keyframes aq-slide { from { background-position: -40% 0; } to { background-position: 140% 0; } }
</style>
"""

# Shown instead of Streamlit's spinner while a dataset variant loads: a wave over a 3 × 3 block of
# "periodic-table cells". Drawn by the page before the (cached) load — a cached function must not
# draw Streamlit elements itself.
LOADING_HTML = """
<div class="aq-loading">
  <div class="aq-grid"><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i></div>
  <div><b>Loading the GNoME screening data</b> — about half a minute<br>
       <span class="aq-stage">Reading the materials table, the decomposition-energy grids (≈1.3 GB)
       and the element index. This happens once per dataset variant.</span></div>
</div>
<style>
.aq-loading { display: flex; align-items: center; gap: 18px; padding: 16px 20px; margin: 6px 0 14px;
              border: 1px solid rgba(128,128,128,.3); border-radius: 8px; }
.aq-grid { display: grid; grid-template-columns: repeat(3, 13px); gap: 4px; flex: none; }
.aq-grid i { display: block; width: 13px; height: 13px; border-radius: 3px; background: #2a7de1;
             animation: aq-pulse 1.4s infinite ease-in-out; }
.aq-grid i:nth-child(2), .aq-grid i:nth-child(4) { animation-delay: .15s; }
.aq-grid i:nth-child(3), .aq-grid i:nth-child(5), .aq-grid i:nth-child(7) { animation-delay: .3s; }
.aq-grid i:nth-child(6), .aq-grid i:nth-child(8) { animation-delay: .45s; }
.aq-grid i:nth-child(9) { animation-delay: .6s; }
@keyframes aq-pulse { 0%, 100% { opacity: .18; transform: scale(.8); }
                      45% { opacity: 1; transform: scale(1); } }
.aq-stage { opacity: .75; font-size: .92em; }
</style>
"""


@st.cache_resource(show_spinner=False)
def _loaded_variants() -> set:
    """Variants already loaded in this process — lets the page skip the loading card on reruns."""
    return set()


@st.cache_resource(max_entries=MAX_VARIANTS, show_spinner=False)
def load_variant(solid_filter: bool, gga_only: bool) -> dict:
    dh = Data_Handler(solid_filter=solid_filter, gga_only=gga_only,
                      path_to_data_directory=DATA_DIR)
    df = dh.get_df().reset_index(drop=True)
    grid = load_decomposition_grid(dh, df)
    membership = element_membership(df)
    n_total = dh.N_total_GNoME
    _loaded_variants().add((solid_filter, gga_only))
    dh.gga_results.close()      # the grids live in memory from here on
    dh.mixed_results.close()
    return {
        "df": df, "grid": grid, "membership": membership, "n_total": n_total,
        "elements": list(membership.columns),
        "element_counts": {e: int(n) for e, n in membership.sum().items()},
        "n_elements": membership.to_numpy().sum(axis=1),
        "dimensionalities": sorted(df['Dimensionality Cheon'].dropna().unique()),
    }


@st.cache_resource(show_spinner=False)
def structure_db():
    return atoms_from_db(DATA_DIR)


def _criteria_from_params(params):
    criteria = [
        Stability_Criteria(Us=list(u) if isinstance(u, tuple) else u,
                           pHs=list(p) if isinstance(p, tuple) else p,
                           decomposition_threshold=t)
        for u, p, t in params
    ]
    # Two criteria over the same region differ only in threshold, but share a column name.
    seen = {}
    for sc in criteria:
        n = seen.get(sc.col_name, 0)
        seen[sc.col_name] = n + 1
        if n:
            sc.col_name = f"{sc.col_name} ({n + 1})"
    return criteria


@st.cache_data(max_entries=16, show_spinner=False)
def run_screen(solid_filter: bool, gga_only: bool, params: tuple):
    bundle = load_variant(solid_filter, gga_only)
    return screen(bundle["grid"], _criteria_from_params(params))


def cif_bytes(row: pd.Series) -> bytes:
    buf = io.BytesIO()      # ASE's CIF writer emits bytes
    ase_write(buf, structure_db().get_atoms_from_row(row), format="cif")
    return buf.getvalue()


def cif_archive(rows: pd.DataFrame) -> bytes:
    """Zip of CIF files for the given result rows (file name = reduced formula, de-duplicated)."""
    buf = io.BytesIO()
    seen = {}
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for _, row in rows.iterrows():
            formula = row['Reduced Formula']
            n = seen.get(formula, 0)
            seen[formula] = n + 1
            name = f"{formula}.cif" if n == 0 else f"{formula}_{row['MaterialId']}.cif"
            zf.writestr(name, cif_bytes(row))
    return buf.getvalue()


def stability_map(grid_row: np.ndarray, criteria, title: str):
    """The material's decomposition energy over the full U × pH grid, with the criterion
    region(s) outlined and the water stability window dashed."""
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    im = ax.imshow(grid_row, origin="lower", aspect="auto", cmap="plasma", vmin=0, vmax=1.0,
                   extent=(PH_MIN - PH_STEP / 2, PH_MAX + PH_STEP / 2,
                           U_MIN - U_STEP / 2, U_MAX + U_STEP / 2))
    fig.colorbar(im, ax=ax, label=r"$\Delta G_{decomp}$ (eV/atom), clipped at 1")
    ph = np.array([PH_MIN, PH_MAX])
    ax.plot(ph, 1.23 - 0.059 * ph, "k--", lw=0.8)
    ax.plot(ph, -0.059 * ph, "k--", lw=0.8)
    for sc in criteria:
        if isinstance(sc.pHs, list):
            p0, p1 = sc.pH_lowlimt_index, sc.pH_highlimt_index
        else:
            p0, p1 = sc.pH_index, sc.pH_index + 1
        if isinstance(sc.Us, list):
            u0, u1 = sc.U_lowlimt_index, sc.U_highlimt_index
        else:
            u0, u1 = sc.U_index, sc.U_index + 1
        ax.add_patch(Rectangle((sc.pH_interval[p0] - PH_STEP / 2, sc.U_interval[u0] - U_STEP / 2),
                               (p1 - p0) * PH_STEP, (u1 - u0) * U_STEP,
                               fill=False, lw=2, ec="black"))
    ax.set_xlabel("pH")
    ax.set_ylabel("U (V vs. SHE)")
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    return fig


def reset_filters():
    st.session_state["pt_nonce"] = st.session_state.get("pt_nonce", 0) + 1
    for k in ["require_mode", *POST_FILTER_KEYS]:
        st.session_state.pop(k, None)


# ── Page ───────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="AQ-GNoME screening (DEMO)", page_icon="⚗️", layout="wide")
st.markdown(PAGE_CSS, unsafe_allow_html=True)
st.title("AQ-GNoME — aqueous stability screening (DEMO)")
st.caption("Pourbaix stability of GNoME crystal structures. Pick a dataset, click elements, set "
           "the operating window.")

# Dataset -----------------------------------------------------------------------
d1, d2, d3 = st.columns([2.2, 1.4, 3], vertical_alignment="bottom")
functional = d1.segmented_control("Functional", list(FUNCTIONALS), key="functional",
                                  default=list(FUNCTIONALS)[0],
                                  help="r2SCAN energies are mixed in via the Materials Project "
                                       "mixing scheme where available.")
gga_only = FUNCTIONALS.get(functional, False)
solid_choice = d2.segmented_control("Solid filter", ["on", "off"], key="solid_filter", default="on",
                                    help="Exclude the material itself from the Pourbaix reference "
                                         "pool, so its decomposition energy is set by competing "
                                         "phases only.")
solid_filter = solid_choice != "off"

loading = st.empty()
if (solid_filter, gga_only) not in _loaded_variants():
    loading.markdown(LOADING_HTML, unsafe_allow_html=True)
bundle = load_variant(solid_filter, gga_only)
loading.empty()
df, grid, membership = bundle["df"], bundle["grid"], bundle["membership"]
d3.caption(f"{bundle['n_total']:,} GNoME materials, {len(df):,} with Pourbaix data in this "
           f"dataset variant.")

# Elements ----------------------------------------------------------------------
with st.container(border=True):
    h1, h2 = st.columns([5, 1], vertical_alignment="center")
    h1.subheader("Filter by element", anchor=False)
    h2.button("Reset filters", on_click=reset_filters, width="stretch")

    nonce = st.session_state.get("pt_nonce", 0)
    pt_key = f"ptable-{nonce}"
    elems = st.session_state.get(f"{pt_key}-elems", {})
    elems = periodic_table(bundle["element_counts"], len(df), elems, key=pt_key)
    st.session_state[f"{pt_key}-elems"] = elems
    require = sorted(e for e, v in elems.items() if v == 1)
    exclude = sorted(e for e, v in elems.items() if v == -1)

    if len(require) > 1:
        require_mode = st.radio("Required elements", ["all of them", "any of them"],
                                horizontal=True, key="require_mode",
                                help="Whether a material must contain every required element or "
                                     "at least one of them.")
    else:
        require_mode = "all of them"

# Stability criteria --------------------------------------------------------------
with st.container(border=True):
    st.subheader("Stability criteria", anchor=False)
    st.caption("A material is stable when its maximum decomposition energy inside the U × pH "
               "window stays at or below the threshold — for every criterion.")
    second = st.toggle("Add a second criterion", key="second_criterion")
    crit_params = []
    for k in range(2 if second else 1):
        c1, c2, c3, c4, c5 = st.columns([1.1, 2.2, 1.1, 2.2, 1.3], vertical_alignment="bottom")
        ph_mode = c1.segmented_control("pH", ["range", "value"], default="range", key=f"ph_mode{k}")
        if ph_mode == "value":
            ph = c2.slider("pH", PH_MIN, PH_MAX, 0.0, PH_STEP, key=f"ph{k}")
        else:
            ph = c2.slider("pH window", PH_MIN, PH_MAX, (0.0, 2.0), PH_STEP, key=f"phr{k}")
        u_mode = c3.segmented_control("Potential", ["range", "value"], default="range",
                                      key=f"u_mode{k}")
        if u_mode == "value":
            u = c4.slider("U (V vs. SHE)", U_MIN, U_MAX, 1.2, U_STEP, key=f"u{k}")
        else:
            u = c4.slider("U window (V vs. SHE)", U_MIN, U_MAX, (1.2, 2.0), U_STEP, key=f"ur{k}")
        thr = c5.number_input("ΔG threshold (eV/atom)", 0.0, 5.0, 0.5, 0.05, key=f"thr{k}")
        crit_params.append((u, ph, float(thr)))
    crit_params = tuple(crit_params)
    criteria = _criteria_from_params(crit_params)
    crit_cols = [sc.col_name for sc in criteria]

# Post-filters ----------------------------------------------------------------------
with st.expander("Post-filters — all off by default"):
    p1, p2 = st.columns(2)
    dim_options = bundle["dimensionalities"] + ["unknown"]
    dims = p1.multiselect("Dimensionality", dim_options, default=dim_options, key="dims")
    n_el = p1.slider("Number of elements", 2, 6, (2, 6), key="n_el")
    nsites_max = int(df['NSites'].max())
    max_nsites = p1.slider("Max. number of sites in the cell", 3, nsites_max, nsites_max,
                           key="max_nsites")
    bandgap = p2.slider("Band gap (eV)", 0.0, BANDGAP_CAP, (0.0, BANDGAP_CAP), 0.05, key="bandgap",
                        help=f"Upper end at {BANDGAP_CAP:g} eV = no upper limit.")
    max_disorder = p2.slider("Max. disorder probability", 0.0, 1.0, 1.0, 0.05, key="max_disorder")
    max_hhi_avg = p2.slider("Max. avg. HHI (production, excl. O/H)", HHI_MIN, HHI_MAX, HHI_MAX,
                            100, key="max_hhi_avg")
    max_hhi_max = p2.slider("Max. HHI (production, any element)", HHI_MIN, HHI_MAX, HHI_MAX, 100,
                            key="max_hhi_max")
    st.caption("Materials without a band-gap or HHI value are dropped when the corresponding "
               "limit is set.")

# ── Screening ──────────────────────────────────────────────────────────────────

elem_ok = element_mask(membership, exclude=exclude, require=require,
                       require_all=(require_mode == "all of them"))
stable_ok, maxes = run_screen(solid_filter, gga_only, crit_params)
n_after_elements = int(elem_ok.sum())
n_stable = int((elem_ok & stable_ok).sum())

result = stable_dataframe(df, elem_ok & stable_ok, criteria, maxes)

# Post-filters only act when moved off their default; a limit on a column with missing values
# (band gap, HHI) drops the rows lacking a value, as the notebook's `df[df[col] < x]` does.
post = pd.Series(True, index=result.index)
post &= result['Dimensionality Cheon'].fillna("unknown").isin(dims)
n_el_result = bundle["n_elements"][result.index.to_numpy()]
post &= (n_el_result >= n_el[0]) & (n_el_result <= n_el[1])
post &= result['NSites'] <= max_nsites
if bandgap[0] > 0.0:
    post &= result['Bandgap'] >= bandgap[0]
if bandgap[1] < BANDGAP_CAP:
    post &= result['Bandgap'] <= bandgap[1]
if max_disorder < 1.0:
    post &= result['Disorder Probability'] <= max_disorder
if max_hhi_avg < HHI_MAX:
    post &= result['average_HHI_P_excluding_O_H'] <= max_hhi_avg
if max_hhi_max < HHI_MAX:
    post &= result['max_HHI_P'] <= max_hhi_max
result = result[post]

# ── Results ────────────────────────────────────────────────────────────────────

st.subheader("Results", anchor=False)
st.markdown(f"**{len(df):,}** with Pourbaix data → **{n_after_elements:,}** after element filters "
            f"→ **{n_stable:,}** stable under the criteria → **{len(result):,}** after post-filters")

if len(result) == 0:
    st.info("No materials match. Loosen the element filters, the criteria or the post-filters.")
    result = None


def render_results(result: pd.DataFrame):
    """Sort/download controls, the results table, and the selected material's detail panel."""
    o1, o2, o3, o4, o5 = st.columns([2, 1, 1.2, 1.6, 1.6], vertical_alignment="bottom")
    sort_col = o1.selectbox("Sort by", crit_cols + ['Bandgap', 'Disorder Probability',
                            'average_HHI_P_excluding_O_H', 'max_HHI_P', 'NSites', 'Reduced Formula'],
                            key="sort_col", format_func=lambda c: COLUMN_LABELS.get(c, c))
    ascending = o2.checkbox("Ascending", value=True, key="ascending")
    max_rows = o3.number_input("Rows shown", 10, 5000, 500, 10, key="max_rows")
    result = result.sort_values(sort_col, ascending=ascending, kind="stable", na_position="last")

    table = get_simplified_df(result)
    table['Elements'] = table['Elements'].apply(", ".join)
    ordered = crit_cols + [c for c in LEAD_COLUMNS if c in table.columns]
    table = table[ordered + [c for c in table.columns if c not in ordered]]
    shown = table.head(int(max_rows))

    o4.download_button("Download CSV (all rows)", table.to_csv(index=False).encode(),
                       "aq_gnome_screening.csv", "text/csv", width="stretch")
    n_cif = min(len(shown), MAX_CIF_ROWS)
    cached = st.session_state.get("cif_zip")
    cif_ids = tuple(shown['MaterialId'].head(n_cif))
    if cached and cached[0] == cif_ids:
        o5.download_button("Download CIF zip", cached[1], "aq_gnome_structures.zip",
                           "application/zip", width="stretch")
    elif o5.button(f"Prepare CIFs ({n_cif})", width="stretch"):
        with st.spinner("Extracting structures …"):
            st.session_state["cif_zip"] = (cif_ids, cif_archive(result.loc[shown.index[:n_cif]]))
        st.rerun()

    if len(table) > len(shown):
        st.caption(f"Showing the first {len(shown):,} of {len(table):,} rows; the CSV holds all rows. "
                   "Click a row to see the material's stability map.")
    else:
        st.caption("Click a row to see the material's stability map.")

    event = st.dataframe(
        shown, hide_index=True, width="stretch", height=480,
        on_select="rerun", selection_mode="single-row", key="results_table",
        column_config={c: st.column_config.NumberColumn(COLUMN_LABELS.get(c, c), format="%.2f")
                       for c in crit_cols + ['Bandgap', 'Disorder Probability']}
                      | {'Dimensionality Cheon': 'Dimensionality'},
    )

    # Selected material ---------------------------------------------------------------------
    selected = event.selection.rows if event and event.selection else []
    if selected:
        row = result.loc[shown.index[selected[0]]]
        st.subheader(f"{row['Reduced Formula']}", anchor=False)
        f1, f2 = st.columns([1.1, 1.4])
        with f1:
            facts = {
                "MaterialId": row['MaterialId'],
                "Composition": row['Composition'],
                "Elements": ", ".join(row['Elements']),
                "Sites in cell": int(row['NSites']),
                "Crystal system": row['Crystal System'],
                "Space group": row.get('Space Group', '—'),
                "Dimensionality": row['Dimensionality Cheon'] if pd.notna(row['Dimensionality Cheon'])
                                  else "unknown",
                "Band gap (eV)": "—" if pd.isna(row['Bandgap']) else f"{row['Bandgap']:.2f}",
                "Disorder probability": f"{row['Disorder Probability']:.2f}",
                "Avg. HHI (production, excl. O/H)": "—" if pd.isna(row['average_HHI_P_excluding_O_H'])
                                                    else f"{row['average_HHI_P_excluding_O_H']:.0f}",
                "Max. HHI (production)": "—" if pd.isna(row['max_HHI_P']) else f"{row['max_HHI_P']:.0f}",
            }
            for c in crit_cols:
                facts[f"Max. ΔG in window — {c}"] = f"{row[c]:.2f} eV/atom"
            st.table(pd.Series(facts, name="").to_frame())
            st.download_button("Download CIF", cif_bytes(row), f"{row['Reduced Formula']}.cif",
                               "chemical/x-cif")
        with f2:
            fig = stability_map(grid[row.name], criteria,
                                f"{row['Reduced Formula']} — decomposition energy")
            st.pyplot(fig)
            plt.close(fig)
            st.caption("Black boxes: the criterion window(s). Dashed lines: water stability window.")


if result is not None:
    render_results(result)

# About -------------------------------------------------------------------------------
with st.expander("About, citation and licence"):
    st.markdown("""
**AQ-GNoME** evaluates the aqueous electrochemical (Pourbaix) stability of crystal structures from
the GNoME database. If you use results from this tool, please cite:

- Nissen, M. S., Beck, P., Karlsson, L., et al. *AQ-GNoME: an Aqueous Stability Augmentation of
  the GNoME Database.* ChemRxiv (2026). https://doi.org/10.26434/chemrxiv.15003677/v1
- **GNoME** — Merchant et al., *Nature* (2023). https://doi.org/10.1038/s41586-023-06735-9
- **Disorder probability** — Jakob et al., *Advanced Materials* (2025).
  https://doi.org/10.1002/adma.202514226
- **HHI scores** — Gaultois et al., *Chem. Mater.* 25, 2911–2920 (2013).
  https://doi.org/10.1021/cm400893e

Code: MIT License, © 2026 Technical University of Denmark —
https://github.com/Matminator/AQ-GNoME. Data: CC BY-NC 4.0 (contains GNoME-derived data,
Google DeepMind) — https://doi.org/10.11583/DTU.30738716. **Non-commercial use only.**
""")
