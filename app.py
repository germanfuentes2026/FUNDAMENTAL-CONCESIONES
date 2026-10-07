"""Fundamental Terminal — Altman Z para empresas que no cotizan (balances en PDF)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import altair as alt
import pandas as pd
import streamlit as st

from src import store
from src.pdf_extract import extract
from src.private_altman import FACTOR_NAMES, FIELDS, MODELS, compute

st.set_page_config(page_title="Fundamental Terminal", page_icon="▣", layout="wide")

# ── Estilo: pegá acá tu bloque BLOOMBERG_CSS original para conservar el look completo ──
st.markdown("""<style>
.stApp{background:radial-gradient(1200px 600px at 10% -10%,#1a2420 0%,#0b0e11 45%,#07090b 100%);color:#d7e0d8}
section[data-testid="stSidebar"]{background:#0a0d10;border-right:1px solid #1f2a22}
.block-container{padding-top:4rem;max-width:1480px}
div[data-testid="stMetric"]{background:#111714;border:1px solid #243328;padding:12px 8px;text-align:center}
.stButton>button{background:#f5a623;color:#111;border:0;font-weight:700;text-transform:uppercase}
.ft-brand{color:#f5a623;font-weight:600;font-size:13px;letter-spacing:.28em;text-transform:uppercase}
.ft-title{font-size:28px;font-weight:700;color:#eef6ef}
</style>""", unsafe_allow_html=True)

if "records" not in st.session_state:
    st.session_state.records = store.load()
records: list[dict] = st.session_state.records


def commit(new_records: list[dict]) -> None:
    st.session_state.records = store.merge(st.session_state.records, new_records)
    store.save(st.session_state.records)


# ───────────────────────── SIDEBAR ─────────────────────────
with st.sidebar:
    st.markdown("**BALANCES**")
    companies = sorted({r["company"] for r in records})
    pick = st.selectbox("Empresa", companies + ["➕ Nueva empresa…"]) if companies else "➕ Nueva empresa…"
    company = st.text_input("Nombre de la empresa").strip() if pick.startswith("➕") else pick
    files = st.file_uploader("Balances (PDF)", type="pdf", accept_multiple_files=True)
    api_key = st.text_input("API key de Anthropic", type="password",
                            value=os.getenv("ANTHROPIC_API_KEY", ""))
    go = st.button("Extraer y agregar", width="stretch")
    st.markdown("---")
    model = st.selectbox("Modelo", list(MODELS), index=1,
                         help="Z'' EM suma 3,25 al Z'' y corre los cortes; es la versión para mercados emergentes.")
    st.markdown("---")
    st.caption("Respaldo de datos")
    st.download_button("Descargar JSON", json.dumps(records, ensure_ascii=False, indent=1),
                       "balances.json", "application/json", width="stretch")
    up = st.file_uploader("Restaurar JSON", type="json", key="restore")
    if up is not None and st.button("Restaurar", width="stretch"):
        commit(json.load(up))
        st.rerun()

if go:
    if not files or not company or not api_key:
        st.sidebar.error("Falta empresa, API key o PDF.")
    else:
        for f in files:
            with st.spinner(f"Leyendo {f.name}…"):
                try:
                    new = extract(f.getvalue(), api_key, f.name, company)
                    commit(new)
                    st.sidebar.success(f"{f.name}: {len(new)} ejercicio(s)")
                except Exception as exc:  # noqa: BLE001
                    st.sidebar.error(f"{f.name}: {exc}")
        st.rerun()

# ───────────────────────── MAIN ─────────────────────────
st.markdown('<div class="ft-brand">Fundamental Terminal</div>'
            '<div class="ft-title">Altman Z-Score · empresas no cotizantes</div>', unsafe_allow_html=True)

mine = [r for r in records if company and r["company"].lower() == company.lower()]
if not mine:
    st.info("Elegí o creá una empresa, subí uno o más balances en PDF y tocá **Extraer y agregar**.")
    st.stop()

rows, comp = [], []
for r in sorted(mine, key=lambda r: r["fy"]):
    c = compute(r, model)
    rows.append({"FY": str(r["fy"]), "period end": r["period_end"], "Z-Score": None if c["z"] is None else round(c["z"], 2),
                 "zone": c["zone"], **{f"X{i + 1}": None if v is None else round(v, 4) for i, v in enumerate(c["x"])},
                 "origen": r.get("origin", "")})
    comp.append(c)
hist = pd.DataFrame(rows)
last = comp[-1]
m = MODELS[model]

c1, c2, c3, c4 = st.columns(4)
c1.metric("EMPRESA", company)
c2.metric("Z-SCORE", "—" if last["z"] is None else f"{last['z']:.2f}", last["zone"])
c3.metric("ÚLTIMO EJERCICIO", hist.iloc[-1]["period end"])
c4.metric("EJERCICIOS CARGADOS", len(hist))

tab_h, tab_c, tab_g, tab_d, tab_f = st.tabs(["HISTORIAL", "COMPONENTES", "CHARTS", "DATOS", "FORMULAS"])

with tab_h:
    st.dataframe(hist, width="stretch", hide_index=True)
    zs = [z for z in hist["Z-Score"] if z is not None]
    if zs:
        st.markdown(f"**Promedio:** `{sum(zs) / len(zs):.2f}` · Mín `{min(zs):.2f}` · Máx `{max(zs):.2f}`")
    st.caption("Los ratios usan valores de una misma columna del balance, por lo que la reexpresión por inflación no los distorsiona.")

with tab_c:
    yr = st.selectbox("Ejercicio", list(hist["FY"])[::-1])
    c = comp[list(hist["FY"]).index(yr)]
    st.dataframe(pd.DataFrame({
        "factor": FACTOR_NAMES, "peso": m["w"],
        "ratio": [None if v is None else round(v, 4) for v in c["x"]],
        "contribución": [None if v is None or not w else round(v, 4) for v, w in zip(c["contrib"], m["w"])],
    }), width="stretch", hide_index=True)
    st.markdown(f"Constante: `{m['const']}` · **Z = {'n/a' if c['z'] is None else f'{c['z']:.3f}'}** · {c['zone']}")
    if c["missing"]:
        st.warning("Faltan datos para: " + ", ".join(c["missing"]))
    notes = [r.get("notes") for r in mine if str(r["fy"]) == yr and r.get("notes")]
    if notes:
        st.caption("Notas de extracción: " + " | ".join(notes))


def style(ch):
    return (ch.configure(background="#0b0e11").configure_view(stroke="#243328")
            .configure_axis(labelColor="#8b9a8d", titleColor="#8b9a8d", gridColor="#1f2a22")
            .configure_legend(labelColor="#8b9a8d", titleColor="#8b9a8d")
            .configure_title(color="#e8f3e9", fontSize=14))


with tab_g:
    d = hist.dropna(subset=["Z-Score"])
    if d.empty:
        st.info("No hay datos suficientes para graficar.")
    else:
        years = list(d["FY"])
        lines = pd.DataFrame([(m["hi"], "SAFE", "#3ddc84"), (m["lo"], "DISTRESS", "#ff5c5c")], columns=["y", "label", "color"])
        rules = alt.Chart(lines).mark_rule(strokeDash=[5, 4]).encode(y="y:Q", color=alt.Color("color:N", scale=None))
        base = alt.Chart(d).encode(x=alt.X("FY:N", sort=years, axis=alt.Axis(labelAngle=0)), y=alt.Y("Z-Score:Q"),
                                   tooltip=["FY", "period end", "Z-Score", "zone"])
        line = base.mark_line(color="#f5a623", strokeWidth=2.5, point=alt.OverlayMarkDef(color="#f5a623", size=90))
        txt = base.mark_text(dy=-14, color="#e8f3e9").encode(text=alt.Text("Z-Score:Q", format=".2f"))
        st.altair_chart(style((rules + line + txt).properties(height=340, title=f"{model}")), width="stretch")

        long = pd.DataFrame([{"FY": h["FY"], "factor": FACTOR_NAMES[i].split()[0], "contribución": c["contrib"][i]}
                             for h, c in zip(rows, comp) if h["Z-Score"] is not None
                             for i in range(5) if m["w"][i] and c["contrib"][i] is not None])
        bars = alt.Chart(long).mark_bar().encode(
            x=alt.X("FY:N", sort=years, axis=alt.Axis(labelAngle=0)), y="contribución:Q",
            color=alt.Color("factor:N", scale=alt.Scale(scheme="tableau10")), tooltip=["FY", "factor", alt.Tooltip("contribución:Q", format=".3f")])
        st.altair_chart(style(bars.properties(height=360, title="Contribución de cada factor al Z")), width="stretch")

with tab_d:
    st.caption("Podés corregir valores, agregar un ejercicio a mano o borrar filas. Luego tocá **Guardar cambios**.")
    cols = ["fy", "period_end", "origin", "source"] + FIELDS + ["notes"]
    df = pd.DataFrame(mine).reindex(columns=cols)
    edited = st.data_editor(df, num_rows="dynamic", width="stretch", hide_index=True,
                            column_config={f: st.column_config.NumberColumn(f, format="%.0f") for f in FIELDS})
    if st.button("Guardar cambios"):
        keep = [r for r in records if r["company"].lower() != company.lower()]
        new = []
        for _, row in edited.dropna(subset=["fy", "period_end"]).iterrows():
            rec = {k: (None if pd.isna(v) else v) for k, v in row.to_dict().items()}
            rec.update(company=company, fy=int(rec["fy"]), origin=rec.get("origin") or "manual")
            new.append(rec)
        st.session_state.records = store.merge(keep, new)
        store.save(st.session_state.records)
        st.rerun()

with tab_f:
    st.markdown(r"""
**X1** = (Activo corriente − Pasivo corriente) / Activo total · **X2** = Resultados acumulados / Activo total ·
**X3** = EBIT / Activo total · **X4** = **Patrimonio neto contable** / Pasivo total · **X5** = Ventas / Activo total

| Modelo | Fórmula | Cortes (distress / safe) |
|---|---|---|
| Z'' | 6,56·X1 + 3,26·X2 + 6,72·X3 + 1,05·X4 | 1,10 / 2,60 |
| Z'' EM | 3,25 + Z'' | 4,15 / 5,85 |
| Z' | 0,717·X1 + 0,847·X2 + 3,107·X3 + 0,420·X4 + 0,998·X5 | 1,23 / 2,90 |

Como la empresa no cotiza, X4 no puede usar capitalización bursátil: se reemplaza por el valor contable del patrimonio.
Resultados acumulados = reservas + resultados no asignados (sin capital ni ajuste de capital). EBIT = resultado operativo antes de resultados financieros e impuestos.
""")







