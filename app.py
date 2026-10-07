"""Fundamental Terminal — Altman Z-Score para empresas que NO cotizan (balances en PDF).

Un solo archivo. Subís balances en PDF, Claude extrae los datos, la app acumula los ejercicios
y arma el historial del Altman Z'' / Z' usando el patrimonio contable en X4.
Requiere: streamlit, pandas, altair, pypdf, anthropic  +  una API key de Anthropic.
"""
from __future__ import annotations

import base64
import io
import json
import math
import os
import re
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

# ═════════════════════════ 1. MODELO ALTMAN ═════════════════════════

FIELDS = [
    "current_assets", "current_liabilities", "total_assets", "total_liabilities",
    "equity", "retained_earnings", "ebit", "sales",
]
FACTOR_NAMES = [
    "X1 Capital de trabajo / Activo",
    "X2 Resultados acumulados / Activo",
    "X3 EBIT / Activo",
    "X4 Patrimonio neto / Pasivo",
    "X5 Ventas / Activo",
]
# w = pesos de X1..X5 (0 = el factor no entra), const = constante, lo/hi = cortes
MODELS = {
    "Z'' (no manufactureras)": dict(w=[6.56, 3.26, 6.72, 1.05, 0.0], const=0.0, lo=1.10, hi=2.60),
    "Z'' EM (mercados emergentes)": dict(w=[6.56, 3.26, 6.72, 1.05, 0.0], const=3.25, lo=4.15, hi=5.85),
    "Z' (privadas manufactureras)": dict(w=[0.717, 0.847, 3.107, 0.420, 0.998], const=0.0, lo=1.23, hi=2.90),
}


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def _div(a, b):
    return None if a is None or b is None or b == 0 else a / b


def compute(rec: dict, model: str) -> dict:
    m = MODELS[model]
    g = {k: _num(rec.get(k)) for k in FIELDS}
    ca, cl = g["current_assets"], g["current_liabilities"]
    wc = None if ca is None or cl is None else ca - cl
    ta, tl = g["total_assets"], g["total_liabilities"]
    x = [_div(wc, ta), _div(g["retained_earnings"], ta), _div(g["ebit"], ta),
         _div(g["equity"], tl), _div(g["sales"], ta)]
    used = [i for i, w in enumerate(m["w"]) if w]
    missing = [f"X{i + 1}" for i in used if x[i] is None]
    contrib = [None if x[i] is None else m["w"][i] * x[i] for i in range(5)]
    z = None if missing else m["const"] + sum(contrib[i] for i in used)
    if z is None:
        zone = "n/a"
    elif z > m["hi"]:
        zone = "SAFE"
    elif z < m["lo"]:
        zone = "DISTRESS"
    else:
        zone = "GREY"
    return {"x": x, "contrib": contrib, "const": m["const"], "z": z, "zone": zone, "missing": missing}


# ═════════════════════════ 2. ALMACENAMIENTO ═════════════════════════

STORE = Path(__file__).resolve().parent / "data" / "balances.json"
_PRIORITY = {"own": 1, "comparative": 0, "manual": 2}


def load() -> list[dict]:
    try:
        return json.loads(STORE.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return []


def save(records: list[dict]) -> None:
    try:
        STORE.parent.mkdir(parents=True, exist_ok=True)
        STORE.write_text(json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")
    except OSError:
        pass  # filesystem de solo lectura (p. ej. hosting): queda en sesión


def merge(records: list[dict], new: list[dict]) -> list[dict]:
    """Un registro por (empresa, FY). Prioridad: manual > balance propio > comparativo."""
    out = {(r["company"].strip().lower(), int(r["fy"])): r for r in records}
    for r in new:
        k = (r["company"].strip().lower(), int(r["fy"]))
        old = out.get(k)
        if old is None or _PRIORITY.get(r.get("origin"), 0) >= _PRIORITY.get(old.get("origin"), 0):
            out[k] = r
    return sorted(out.values(), key=lambda r: (r["company"].lower(), int(r["fy"])))


# ═════════════════════════ 3. EXTRACCIÓN DESDE PDF ═════════════════════════

MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
MAX_PAGES = 90  # la API procesa hasta ~100 páginas por PDF
KEYWORDS = re.compile(
    r"estado de situaci[oó]n|balance general|estado de resultados|estado del resultado|"
    r"evoluci[oó]n del patrimonio|total del activo|total activo", re.I)

PROMPT = """Sos un analista contable argentino. Del PDF adjunto (estados contables de una concesionaria vial)
extraé los datos para calcular el Altman Z-Score. Devolvé SOLO un JSON, sin texto ni markdown, con esta forma:

{"company": "razón social",
 "periods": [
  {"period_end": "YYYY-MM-DD", "column": "current" | "prior",
   "current_assets": n, "current_liabilities": n, "total_assets": n, "total_liabilities": n,
   "equity": n, "retained_earnings": n, "ebit": n, "sales": n, "notes": "texto breve"}
 ]}

Reglas:
- Una entrada por cada columna del estado de situación patrimonial: el ejercicio actual ("current") y el comparativo ("prior").
- Números en unidades completas (si el estado dice "en millones", multiplicá por 1.000.000). Pérdidas con signo negativo.
- retained_earnings = reservas (legal, facultativa, otras) + resultados no asignados/acumulados (incluye el resultado del ejercicio). NO incluyas capital social ni ajuste de capital.
- ebit = resultado operativo antes de resultados financieros (intereses, diferencias de cambio, RECPAM, tenencia) y antes de impuesto a las ganancias.
  Si no hay esa línea, calculalo como resultado antes de impuesto menos resultados financieros netos. En "notes" aclará cómo lo obtuviste.
- sales = ingresos operativos (peajes / concesión).
- Si un dato no figura, usá null. No inventes valores."""


def select_pages(pdf_bytes: bytes) -> bytes:
    """Si el PDF es muy largo, deja solo las páginas de estados contables."""
    from pypdf import PdfReader, PdfWriter

    reader = PdfReader(io.BytesIO(pdf_bytes))
    n = len(reader.pages)
    if n <= MAX_PAGES:
        return pdf_bytes
    hits = []
    for i, p in enumerate(reader.pages):
        try:
            if KEYWORDS.search(p.extract_text() or ""):
                hits.append(i)
        except Exception:  # noqa: BLE001
            continue
    keep = sorted({j for i in hits for j in (i, i + 1) if j < n})[:MAX_PAGES] or list(range(MAX_PAGES))
    w = PdfWriter()
    for i in keep:
        w.add_page(reader.pages[i])
    buf = io.BytesIO()
    w.write(buf)
    return buf.getvalue()


def parse_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    a, b = text.find("{"), text.rfind("}")
    if a < 0 or b < 0:
        raise ValueError("La respuesta no contiene un JSON.")
    return json.loads(text[a:b + 1])


def normalize(result: dict, source: str, company: str | None = None) -> list[dict]:
    recs = []
    for p in result.get("periods", []):
        end = str(p.get("period_end") or "")[:10]
        if not re.match(r"\d{4}-\d{2}-\d{2}$", end):
            continue
        recs.append({
            "company": company or result.get("company") or "Sin nombre",
            "fy": int(end[:4]), "period_end": end,
            "origin": "own" if p.get("column") == "current" else "comparative",
            "source": source,
            **{k: p.get(k) for k in ("current_assets", "current_liabilities", "total_assets",
                                      "total_liabilities", "equity", "retained_earnings", "ebit", "sales")},
            "notes": p.get("notes") or "",
        })
    return recs


def extract(pdf_bytes: bytes, api_key: str, source: str, company: str | None = None, model: str = MODEL) -> list[dict]:
    import anthropic

    data = base64.standard_b64encode(select_pages(pdf_bytes)).decode()
    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model=model, max_tokens=3000,
        messages=[{"role": "user", "content": [
            {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": data}},
            {"type": "text", "text": PROMPT}]}],
    )
    text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    return normalize(parse_json(text), source, company)


# ═════════════════════════ 4. APP STREAMLIT ═════════════════════════
def _secret(name: str) -> str:
    try:
        return str(st.secrets.get(name, "") or "")
    except Exception:  # noqa: BLE001  (no hay secrets.toml)
        return ""


st.set_page_config(page_title="Fundamental Terminal", page_icon="▣", layout="wide")

BLOOMBERG_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: "IBM Plex Sans", "Segoe UI", sans-serif; }
.stApp { background: radial-gradient(1200px 600px at 10% -10%, #1a2420 0%, #0b0e11 45%, #07090b 100%); color: #d7e0d8; }
header[data-testid="stHeader"] { background: #0b0e11; border-bottom: 1px solid #1f2a22; }
section[data-testid="stSidebar"] { background: #0a0d10; border-right: 1px solid #1f2a22; }
section[data-testid="stSidebar"] * { color: #c5d0c6; }
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] [data-baseweb="select"] > div,
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] { background: #111714 !important; border-color: #243328 !important; }
.block-container { padding-top: 4.5rem; max-width: 1480px; }
h1, h2, h3 { font-family: "IBM Plex Sans", sans-serif; letter-spacing: 0.04em; }
.ft-masthead { display: flex; justify-content: space-between; align-items: flex-end; border: 1px solid #243328;
  background: linear-gradient(90deg, #101612 0%, #0d1210 60%, #151208 100%); padding: 14px 18px 12px 18px; margin-bottom: 14px; }
.ft-brand { font-family: "IBM Plex Mono", monospace; color: #f5a623; font-weight: 600; font-size: 13px;
  letter-spacing: 0.28em; text-transform: uppercase; line-height: 1.4; }
.ft-title { font-size: 28px; font-weight: 700; color: #eef6ef; line-height: 1.1; margin-top: 4px; }
.ft-sub { color: #7f8f82; font-size: 13px; margin-top: 4px; }
.ft-clock { text-align: right; font-family: "IBM Plex Mono", monospace; color: #9aa89b; font-size: 12px; }
.ft-ticker { color: #f5a623; font-size: 22px; font-weight: 600; }
div[data-testid="stMetric"] { background: #111714; border: 1px solid #243328; padding: 12px 8px; text-align: center;
  display: flex; flex-direction: column; align-items: center; justify-content: center; }
div[data-testid="stMetric"] [data-testid="stMetricLabel"],
div[data-testid="stMetric"] [data-testid="stMetricValue"],
div[data-testid="stMetric"] [data-testid="stMetricDelta"] { width: 100%; justify-content: center; text-align: center; }
div[data-testid="stMetric"] [data-testid="stMetricLabel"] *,
div[data-testid="stMetric"] [data-testid="stMetricValue"] *,
div[data-testid="stMetric"] [data-testid="stMetricDelta"] * { white-space: normal !important; overflow: visible !important;
  text-overflow: clip !important; text-align: center; justify-content: center; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] > div { font-size: 1.7rem; line-height: 1.2; }
div[data-testid="stMetric"] label { color: #8b9a8d !important; font-family: "IBM Plex Mono", monospace;
  letter-spacing: 0.12em; font-size: 11px !important; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] { font-family: "IBM Plex Mono", monospace; color: #e8f3e9; }
.stTabs [data-baseweb="tab-list"] { gap: 4px; background: #0d1110; border-bottom: 1px solid #243328; }
.stTabs [data-baseweb="tab"] { background: #0d1110; color: #8b9a8d; font-family: "IBM Plex Mono", monospace; letter-spacing: 0.08em; }
.stTabs [aria-selected="true"] { color: #f5a623 !important; border-bottom: 2px solid #f5a623; }
.stButton>button { background: #f5a623; color: #111; border: 0; font-weight: 700; letter-spacing: 0.08em;
  text-transform: uppercase; font-family: "IBM Plex Mono", monospace; }
.stButton>button:hover { background: #ffc056; color: #111; }
hr { border-color: #243328; }
.stDataFrame { border: 1px solid #243328; }
</style>
"""
st.markdown(BLOOMBERG_CSS, unsafe_allow_html=True)


def _fmt_num(v) -> str:
    if v is None or pd.isna(v):
        return "—"
    for lim, suf in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if abs(v) >= lim:
            return f"{v / lim:,.2f}{suf}"
    return f"{v:,.0f}"


def render_masthead(company: str | None, model: str, stmt: str = "") -> None:
    tk = f'<div class="ft-ticker">{company}</div>' if company else ""
    stmt_html = f"<br/>{stmt}" if stmt else ""
    st.markdown(
        f"""<div class="ft-masthead"><div>
        <div class="ft-brand">Fundamental Terminal</div>
        <div class="ft-title">Altman Z-Score · Empresas no cotizantes</div>
        <div class="ft-sub">{model} · patrimonio contable en X4</div>{tk}</div>
        <div class="ft-clock">Data: balances cargados (PDF){stmt_html}</div></div>""",
        unsafe_allow_html=True)


def _style_chart(ch):
    return (ch.configure(background="#0b0e11").configure_view(stroke="#243328")
            .configure_axis(labelColor="#8b9a8d", titleColor="#8b9a8d", gridColor="#1f2a22", domainColor="#243328",
                            tickColor="#243328", labelFont="IBM Plex Mono", titleFont="IBM Plex Mono")
            .configure_legend(labelColor="#8b9a8d", titleColor="#8b9a8d")
            .configure_title(color="#e8f3e9", font="IBM Plex Mono", fontSize=14))


def _threshold_layer(levels):
    df = pd.DataFrame(levels, columns=["y", "label", "color"])
    rules = alt.Chart(df).mark_rule(strokeDash=[5, 4], opacity=0.8).encode(y="y:Q", color=alt.Color("color:N", scale=None))
    texts = alt.Chart(df).mark_text(align="left", dx=4, dy=-6, fontSize=10).encode(
        y="y:Q", text="label:N", color=alt.Color("color:N", scale=None), x=alt.value(4))
    return rules + texts

if "records" not in st.session_state:
    st.session_state.records = load()
records: list[dict] = st.session_state.records


def commit(new_records: list[dict]) -> None:
    st.session_state.records = merge(st.session_state.records, new_records)
    save(st.session_state.records)


# ───────────────────────── SIDEBAR ─────────────────────────
if "next_pick" in st.session_state:  # selecciona la empresa recién cargada
    st.session_state["pick_company"] = st.session_state.pop("next_pick")

with st.sidebar:
    st.markdown("**COMMAND**")
    companies = sorted({r["company"] for r in records})
    pick = (st.selectbox("Empresa", companies + ["➕ Nueva empresa…"], key="pick_company")
            if companies else "➕ Nueva empresa…")
    company = st.text_input("Nombre de la empresa").strip() if pick.startswith("➕") else pick
    files = st.file_uploader("Balances (PDF)", type="pdf", accept_multiple_files=True)
    api_key = st.text_input("API key de Anthropic", type="password",
                            value=os.getenv("ANTHROPIC_API_KEY", "") or _secret("ANTHROPIC_API_KEY"))
    with st.expander("Avanzado"):
        ext_model = st.text_input("Modelo de extracción", value=MODEL,
                                  help="Si da error de modelo, probá con claude-sonnet-4-6.")
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
    flash = []
    if not files or not company or not api_key:
        flash.append(("error", "Falta el nombre de la empresa, la API key o algún PDF."))
    else:
        for f in files:
            with st.spinner(f"Leyendo {f.name}… (puede tardar 30–90 s)"):
                try:
                    new = extract(f.getvalue(), api_key, f.name, company, ext_model)
                    if new:
                        commit(new)
                        flash.append(("success", f"{f.name}: {len(new)} ejercicio(s) cargado(s)."))
                    else:
                        flash.append(("warning", f"{f.name}: no se detectaron ejercicios."))
                except Exception as exc:  # noqa: BLE001
                    flash.append(("error", f"{f.name}: {exc}"))
        st.session_state["next_pick"] = company
    st.session_state["flash"] = flash
    st.rerun()

for _kind, _msg in st.session_state.pop("flash", []):
    getattr(st.sidebar, _kind)(_msg)

with st.sidebar:
    st.markdown("---")
    st.markdown("**ABOUT THE MODELS**")
    st.caption("Altman Z'' (1995): versión para empresas no manufactureras y mercados emergentes. "
               "Z = 6,56 X1 + 3,26 X2 + 6,72 X3 + 1,05 X4. Cortes 1,10 / 2,60 (EM: +3,25 y 4,15 / 5,85). "
               "X4 usa patrimonio contable porque la empresa no cotiza. No es una calificación crediticia.")
    st.caption("Educational desk. Verificá los datos extraídos en la pestaña DATOS.")

# ───────────────────────── MAIN ─────────────────────────
mine = [r for r in records if company and r["company"].lower() == company.lower()]
if not mine:
    render_masthead(company or None, model)
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
latest = sorted(mine, key=lambda r: r["fy"])[-1]
render_masthead(company, model, f"Financials: FY{latest['fy']} · period end {latest['period_end']}")

_dc = {"SAFE": "normal", "GREY": "off", "DISTRESS": "inverse"}.get(last["zone"], "off")
zval = "—" if last["z"] is None else f"{last['z']:.2f}"
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("EMPRESA", company)
c2.metric("Z-SCORE", zval, last["zone"], delta_color=_dc)
c3.metric("ACTIVO TOTAL", _fmt_num(latest.get("total_assets")))
c4.metric("PATRIMONIO NETO", _fmt_num(latest.get("equity")))
c5.metric("EJERCICIOS", len(hist))
c6.metric("FY", str(latest["fy"]))

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
    ztxt = "n/a" if c["z"] is None else format(c["z"], ".3f")
    st.markdown(f"Constante: `{m['const']}` · **Z = {ztxt}** · {c['zone']}")
    if c["missing"]:
        st.warning("Faltan datos para: " + ", ".join(c["missing"]))
    notes = [r.get("notes") for r in mine if str(r["fy"]) == yr and r.get("notes")]
    if notes:
        st.caption("Notas de extracción: " + " | ".join(notes))



with tab_g:
    st.subheader("Evolución de los indicadores")
    st.caption("Ejercicios cargados, del más antiguo al más reciente.")
    d = hist.dropna(subset=["Z-Score"])
    if d.empty:
        st.info("No hay datos suficientes para graficar.")
    else:
        years = list(d["FY"])
        lo = min(m["lo"], float(d["Z-Score"].min())) - 0.5
        hi = max(m["hi"], float(d["Z-Score"].max())) + 0.5
        base = alt.Chart(d).encode(
            x=alt.X("FY:N", title="Fiscal year", sort=years, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Z-Score:Q", title="Z-Score", scale=alt.Scale(domain=[lo, hi])),
            tooltip=["FY", "period end", "Z-Score", "zone"])
        line = base.mark_line(color="#f5a623", strokeWidth=2.5,
                              point=alt.OverlayMarkDef(color="#f5a623", size=90, filled=True))
        labels = base.mark_text(dy=-14, color="#e8f3e9", fontSize=12, font="IBM Plex Mono").encode(
            text=alt.Text("Z-Score:Q", format=".2f"))
        zones = _threshold_layer([(m["hi"], f"SAFE > {m['hi']}", "#3ddc84"),
                                  (m["lo"], f"DISTRESS < {m['lo']}", "#ff5c5c")])
        st.altair_chart(_style_chart((zones + line + labels).properties(height=340, title=model)), width="stretch")

        long = pd.DataFrame([
            {"FY": h["FY"], "factor": FACTOR_NAMES[i].split()[0], "contribución": c["contrib"][i]}
            for h, c in zip(rows, comp) if h["Z-Score"] is not None
            for i in range(5) if m["w"][i] and c["contrib"][i] is not None])
        bars = alt.Chart(long).mark_bar(stroke="#0b0e11", strokeWidth=1).encode(
            x=alt.X("FY:N", title="Fiscal year", sort=years, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("contribución:Q", title="Contribución al Z-Score"),
            color=alt.Color("factor:N", title="Factor", scale=alt.Scale(scheme="tableau10")),
            tooltip=["FY", "factor", alt.Tooltip("contribución:Q", format=".3f")])
        total = alt.Chart(d).mark_point(shape="diamond", size=140, filled=True, color="#f5a623", stroke="#0b0e11").encode(
            x=alt.X("FY:N", sort=years), y="Z-Score:Q", tooltip=["FY", alt.Tooltip("Z-Score:Q", format=".2f"), "zone"])
        total_lbl = alt.Chart(d).mark_text(dx=22, color="#f5a623", fontSize=12, font="IBM Plex Mono").encode(
            x=alt.X("FY:N", sort=years), y="Z-Score:Q", text=alt.Text("Z-Score:Q", format=".2f"))
        st.altair_chart(_style_chart((zones + bars + total + total_lbl).properties(
            height=380, title="Contribución de X1–X5 por año")), width="stretch")
        st.caption("Barras: peso × ratio de cada factor (las negativas bajan el Z). El rombo ámbar es el Z-Score total.")

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
        st.session_state.records = merge(keep, new)
        save(st.session_state.records)
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








