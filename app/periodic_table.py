"""Clickable periodic table for element filtering — a Streamlit components-v2 component.

Interaction follows the OER-GNoME database page: click an element once to *require* it, twice to
*exclude* it, a third time to clear. Untouched cells are tinted by how often the element occurs in
the dataset; elements absent from the dataset are greyed out.

Every click is reported back through ``setStateValue('elems', …)``; Python reads it from the
component result in the same run. The ``data`` passed in only seeds a freshly mounted instance,
so changing the ``key`` resets the selection.
"""
import streamlit as st

# Symbol, grid column, grid row (lanthanides / actinides on rows 9 / 10 below a spacer row).
_LAYOUT = [
    ("H", 1, 1), ("He", 18, 1),
    ("Li", 1, 2), ("Be", 2, 2), ("B", 13, 2), ("C", 14, 2), ("N", 15, 2), ("O", 16, 2),
    ("F", 17, 2), ("Ne", 18, 2),
    ("Na", 1, 3), ("Mg", 2, 3), ("Al", 13, 3), ("Si", 14, 3), ("P", 15, 3), ("S", 16, 3),
    ("Cl", 17, 3), ("Ar", 18, 3),
    ("K", 1, 4), ("Ca", 2, 4), ("Sc", 3, 4), ("Ti", 4, 4), ("V", 5, 4), ("Cr", 6, 4),
    ("Mn", 7, 4), ("Fe", 8, 4), ("Co", 9, 4), ("Ni", 10, 4), ("Cu", 11, 4), ("Zn", 12, 4),
    ("Ga", 13, 4), ("Ge", 14, 4), ("As", 15, 4), ("Se", 16, 4), ("Br", 17, 4), ("Kr", 18, 4),
    ("Rb", 1, 5), ("Sr", 2, 5), ("Y", 3, 5), ("Zr", 4, 5), ("Nb", 5, 5), ("Mo", 6, 5),
    ("Tc", 7, 5), ("Ru", 8, 5), ("Rh", 9, 5), ("Pd", 10, 5), ("Ag", 11, 5), ("Cd", 12, 5),
    ("In", 13, 5), ("Sn", 14, 5), ("Sb", 15, 5), ("Te", 16, 5), ("I", 17, 5), ("Xe", 18, 5),
    ("Cs", 1, 6), ("Ba", 2, 6), ("Hf", 4, 6), ("Ta", 5, 6), ("W", 6, 6), ("Re", 7, 6),
    ("Os", 8, 6), ("Ir", 9, 6), ("Pt", 10, 6), ("Au", 11, 6), ("Hg", 12, 6), ("Tl", 13, 6),
    ("Pb", 14, 6), ("Bi", 15, 6), ("Po", 16, 6), ("At", 17, 6), ("Rn", 18, 6),
    ("Fr", 1, 7), ("Ra", 2, 7), ("Rf", 4, 7), ("Db", 5, 7), ("Sg", 6, 7), ("Bh", 7, 7),
    ("Hs", 8, 7), ("Mt", 9, 7), ("Ds", 10, 7), ("Rg", 11, 7), ("Cn", 12, 7), ("Nh", 13, 7),
    ("Fl", 14, 7), ("Mc", 15, 7), ("Lv", 16, 7), ("Ts", 17, 7), ("Og", 18, 7),
    ("La", 3, 9), ("Ce", 4, 9), ("Pr", 5, 9), ("Nd", 6, 9), ("Pm", 7, 9), ("Sm", 8, 9),
    ("Eu", 9, 9), ("Gd", 10, 9), ("Tb", 11, 9), ("Dy", 12, 9), ("Ho", 13, 9), ("Er", 14, 9),
    ("Tm", 15, 9), ("Yb", 16, 9), ("Lu", 17, 9),
    ("Ac", 3, 10), ("Th", 4, 10), ("Pa", 5, 10), ("U", 6, 10), ("Np", 7, 10), ("Pu", 8, 10),
    ("Am", 9, 10), ("Cm", 10, 10), ("Bk", 11, 10), ("Cf", 12, 10), ("Es", 13, 10),
    ("Fm", 14, 10), ("Md", 15, 10), ("No", 16, 10), ("Lr", 17, 10),
]

_HTML = """
<div class="ptblock">
  <div class="ptbar">
    <span class="pthint">Click once to <b>require</b> &middot; twice to <b>exclude</b> &middot;
      a third time to clear. Shading shows how often each element occurs.</span>
    <span class="ptbadge"></span>
  </div>
  <div class="ptscroll"><div class="ptable" role="group" aria-label="Filter by element"></div></div>
  <div class="ptsummary"></div>
</div>
"""

_CSS = """
.ptblock { display: flex; flex-direction: column; gap: 10px; font-family: var(--st-font, sans-serif);
           color: var(--st-text-color); }
.ptbar { display: flex; align-items: baseline; justify-content: space-between; gap: 10px 18px;
         flex-wrap: wrap; }
.pthint { font-size: .78rem; opacity: .75; }
.pthint b { font-weight: 600; }
.ptbadge { font-size: .78rem; font-weight: 600; color: var(--st-primary-color); }
.ptscroll { overflow-x: auto; padding-bottom: 2px; }
.ptable { display: grid; grid-template-columns: repeat(18, 1fr);
          grid-template-rows: repeat(7, auto) 10px auto auto; gap: 3px; min-width: 640px; }
.el { grid-column: var(--c); grid-row: var(--r); aspect-ratio: 1 / 0.9; min-width: 0; padding: 0;
      border: 1px solid var(--st-border-color, #d6d6d6); border-radius: 4px;
      background: var(--st-secondary-background-color); color: var(--st-text-color);
      font: inherit; font-size: .8rem; line-height: 1.1; cursor: pointer;
      display: flex; flex-direction: column; align-items: center; justify-content: center;
      gap: 1px; transition: border-color .1s, transform .05s; }
.el:active { transform: scale(.96); }
.el .sym { font-weight: 500; }
.el .cnt { font-size: .58em; opacity: .6; font-variant-numeric: tabular-nums; }
.el.req .cnt, .el.exc .cnt { opacity: .85; }
.el.off { opacity: .2; cursor: default; }
.el:not(.off):hover { border-color: var(--st-text-color); }
.el.req { background: var(--st-primary-color); border-color: var(--st-primary-color); color: #fff;
          font-weight: 700; }
.el.exc { background: color-mix(in srgb, #d14 14%, var(--st-secondary-background-color));
          border-color: #d14; color: #d14; font-weight: 700; }
.el.exc .sym { text-decoration: line-through; }
.ptsummary { font-size: .8rem; display: flex; flex-wrap: wrap; gap: 6px 14px; align-items: center;
             min-height: 22px; }
.ptsummary .muted { opacity: .6; }
.ptsummary .tagset { display: inline-flex; gap: 5px; flex-wrap: wrap; align-items: center; }
.ptsummary .etag { font-family: var(--st-code-font, monospace); font-size: .74rem; padding: 2px 7px;
                   border-radius: 4px; border: 1px solid; }
.ptsummary .etag.req { color: var(--st-primary-color);
                       border-color: color-mix(in srgb, var(--st-primary-color) 45%, transparent);
                       background: color-mix(in srgb, var(--st-primary-color) 12%, transparent); }
.ptsummary .etag.exc { color: #d14; border-color: color-mix(in srgb, #d14 45%, transparent);
                       background: color-mix(in srgb, #d14 10%, transparent); }
"""

_JS = """
export default function (component) {
  const { data, setStateValue, parentElement } = component;
  const root = parentElement.querySelector('.ptable');
  const badge = parentElement.querySelector('.ptbadge');
  const summary = parentElement.querySelector('.ptsummary');
  const LAYOUT = data.layout;
  const counts = data.counts || {};
  const total = data.total || 0;
  const max = Math.max(1, ...Object.values(counts));
  // symbol -> 1 (require) | -1 (exclude). The host (shadow root) outlives re-renders triggered by
  // new data, so the live selection is kept there and data.elems only seeds a fresh instance.
  const elems = parentElement.__aqElems || Object.assign({}, data.elems || {});
  parentElement.__aqElems = elems;
  const WORD = { 1: 'required', '-1': 'excluded' };
  const fmt = n => n >= 10000 ? Math.round(n / 1000) + 'k' : n >= 1000 ? (n / 1000).toFixed(1) + 'k' : String(n);

  function render() {
    root.innerHTML = LAYOUT.map(([sym, c, r]) => {
      const n = counts[sym] || 0;
      const m = elems[sym] || 0;
      if (!n) {
        return `<button class="el off" style="--c:${c};--r:${r}" disabled
                 title="${sym} — not in this dataset"><span class="sym">${sym}</span></button>`;
      }
      const cls = 'el' + (m === 1 ? ' req' : m === -1 ? ' exc' : '');
      const tint = m ? '' :
        `background:color-mix(in srgb, var(--st-text-color) ${Math.round(n / max * 26)}%, var(--st-secondary-background-color));`;
      const note = m ? ' · ' + WORD[m] : '';
      return `<button class="${cls}" style="--c:${c};--r:${r};${tint}" data-el="${sym}"
               title="${sym} — in ${n.toLocaleString()} of ${total.toLocaleString()} materials${note}">
               <span class="sym">${sym}</span><span class="cnt">${fmt(n)}</span></button>`;
    }).join('');

    const req = Object.keys(elems).filter(s => elems[s] === 1);
    const exc = Object.keys(elems).filter(s => elems[s] === -1);
    const nExc = exc.length;
    badge.textContent = [req.length ? req.length + ' required' : '', nExc ? nExc + ' excluded' : '']
      .filter(Boolean).join(', ');
    const tag = (s, cls) => `<span class="etag ${cls}">${s}</span>`;
    summary.innerHTML =
      (req.length ? `<span class="tagset"><b>Required</b> ${req.map(s => tag(s, 'req')).join('')}</span>` : '') +
      (nExc ? `<span class="tagset"><b>Excluded</b> ${exc.map(s => tag(s, 'exc')).join('')}</span>` : '') ||
      '<span class="muted">No element filter — click elements above.</span>';
  }

  function onClick(e) {
    const b = e.target.closest('button.el');
    if (!b || b.disabled) return;
    const sym = b.dataset.el;
    const cur = elems[sym] || 0;
    const next = cur === 0 ? 1 : cur === 1 ? -1 : 0;
    if (next === 0) delete elems[sym]; else elems[sym] = next;
    render();
    setStateValue('elems', Object.assign({}, elems));
  }

  render();
  // Streamlit re-invokes this function whenever `data` changes, without running the previous
  // cleanup first. Assigning (not adding) the handler keeps exactly one listener alive.
  root.onclick = onClick;
  return () => { root.onclick = null; };
}
"""

_component = st.components.v2.component("aq_gnome_periodic_table", html=_HTML, css=_CSS, js=_JS)


def periodic_table(counts: dict, total: int, elems: dict, key: str) -> dict:
    """Render the table and return the clicked selection ``{symbol: 1 | -1}``.

    ``elems`` seeds a freshly mounted instance (normally the previous return value). Change
    ``key`` to reset the selection.
    """
    result = _component(
        data={"layout": _LAYOUT, "counts": counts, "total": total, "elems": elems},
        default={"elems": {}},
        key=key,
        on_elems_change=lambda: None,
    )
    return dict(result.elems or {})
