"""Fundamental Terminal — Altman Z-Score + Piotroski F-Score para concesionarias viales que NO cotizan.

Subís los balances en PDF, Claude los lee (la API key se configura UNA vez en el servidor, no aparece en la pantalla),
la app acumula los ejercicios y calcula el historial de Altman y Piotroski.
Requiere: streamlit, pandas, altair, pypdf, anthropic
API key: variable de entorno ANTHROPIC_API_KEY o .streamlit/secrets.toml  ->  ANTHROPIC_API_KEY = "sk-ant-..."
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
from html import escape

# ═════════════════════════ 1. MODELOS ═════════════════════════

FIELDS = [
    "current_assets", "current_liabilities", "total_assets", "total_liabilities",
    "equity", "retained_earnings", "ebit", "sales",
    # extra para Piotroski
    "net_income", "cfo", "long_term_debt", "cost_of_sales", "share_capital",
]
FACTOR_NAMES = [
    "X1 Capital de trabajo / Activo",
    "X2 Resultados acumulados / Activo",
    "X3 EBIT / Activo",
    "X4 Patrimonio neto / Pasivo",
    "X5 Ventas / Activo",
]
MODELS = {
    "Z'' (no manufactureras)": dict(w=[6.56, 3.26, 6.72, 1.05, 0.0], const=0.0, lo=1.10, hi=2.60),
    "Z'' EM (mercados emergentes)": dict(w=[6.56, 3.26, 6.72, 1.05, 0.0], const=3.25, lo=4.15, hi=5.85),
    "Z' (privadas manufactureras)": dict(w=[0.717, 0.847, 3.107, 0.420, 0.998], const=0.0, lo=1.23, hi=2.90),
}
PIO_NAMES = [
    "1 Resultado neto > 0",
    "2 Flujo operativo (CFO) > 0",
    "3 ROA sube vs. año anterior",
    "4 CFO > Resultado neto (calidad de ganancias)",
    "5 Deuda LP / Activo baja",
    "6 Liquidez corriente sube",
    "7 Sin aumento de capital social",
    "8 Margen bruto sube",
    "9 Rotación del activo sube",
]


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


def piotroski(cur: dict, prev: dict | None) -> dict:
    """F-Score de 9 criterios. Compara RATIOS de cada año (la inflación no los distorsiona).
    Sin año anterior cargado, los criterios de variación quedan n/d."""
    def g(r, k):
        return _num(r.get(k)) if r else None

    roa = lambda r: _div(g(r, "net_income"), g(r, "total_assets"))
    lev = lambda r: _div(g(r, "long_term_debt"), g(r, "total_assets"))
    cr = lambda r: _div(g(r, "current_assets"), g(r, "current_liabilities"))
    at = lambda r: _div(g(r, "sales"), g(r, "total_assets"))

    def gm(r):
        s_, c_ = g(r, "sales"), g(r, "cost_of_sales")
        return None if s_ in (None, 0) or c_ is None else (s_ - abs(c_)) / s_

    def dlt(f):
        x, y = f(cur), f(prev) if prev else None
        return None if x is None or y is None else x - y

    pct = lambda v: "n/d" if v is None else f"{v * 100:.2f}%"
    num = lambda v: "n/d" if v is None else f"{v:.2f}"

    def big(v):
        if v is None:
            return "n/d"
        for lim, suf in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
            if abs(v) >= lim:
                return f"{v / lim:,.2f}{suf}"
        return f"{v:,.0f}"

    ni, cfo = g(cur, "net_income"), g(cur, "cfo")
    d_roa, d_lev, d_cr, d_gm, d_at = dlt(roa), dlt(lev), dlt(cr), dlt(gm), dlt(at)
    sc_c, sc_p = g(cur, "share_capital"), (g(prev, "share_capital") if prev else None)
    spec = [
        ("F_ROA", "Positive ROA", None if ni is None else ni > 0, f"ROA_t = {pct(roa(cur))}"),
        ("F_CFO", "Positive operating cash flow", None if cfo is None else cfo > 0, f"CFO_t = {big(cfo)}"),
        ("F_dROA", "ROA improving", None if d_roa is None else d_roa > 0,
         f"ROA_t = {pct(roa(cur))} vs ROA_t-1 = {pct(roa(prev) if prev else None)}"),
        ("F_ACCRUAL", "CFO > Net income", None if ni is None or cfo is None else cfo > ni,
         f"CFO = {big(cfo)} · NI = {big(ni)}"),
        ("F_dLEVER", "Lower LT debt / assets", None if d_lev is None else (d_lev < 0 or lev(cur) == 0),
         f"LTD/TA_t = {pct(lev(cur))} vs LTD/TA_t-1 = {pct(lev(prev) if prev else None)}"),
        ("F_dLIQUID", "Higher current ratio", None if d_cr is None else d_cr > 0,
         f"CR_t = {num(cr(cur))} vs CR_t-1 = {num(cr(prev) if prev else None)}"),
        ("F_EQ_OFFER", "No new equity issued", None if sc_c is None or sc_p is None else sc_c <= sc_p * 1.0001,
         f"Capital social_t = {big(sc_c)} vs t-1 = {big(sc_p)}"),
        ("F_dMARGIN", "Higher gross margin", None if d_gm is None else d_gm > 0,
         f"GM_t = {pct(gm(cur))} vs GM_t-1 = {pct(gm(prev) if prev else None)}"),
        ("F_dTURN", "Higher asset turnover", None if d_at is None else d_at > 0,
         f"AT_t = {num(at(cur))} vs AT_t-1 = {num(at(prev) if prev else None)}"),
    ]
    tests = [t[2] for t in spec]
    n = sum(t is not None for t in tests)
    score = sum(bool(t) for t in tests if t is not None)
    if n == 0:
        zone = "n/a"
    elif score >= 8:
        zone = "HIGH QUALITY"
    elif score >= 4:
        zone = "MEDIUM"
    else:
        zone = "LOW QUALITY"
    return {"tests": tests, "spec": spec, "score": None if n == 0 else score, "n": n, "zone": zone}


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
    """Un registro por (empresa, FY). Prioridad: manual > balance propio > comparativo.
    Si el registro ganador no tiene un campo y el otro sí, se completa."""
    out = {(r["company"].strip().lower(), int(r["fy"])): r for r in records}
    for r in new:
        k = (r["company"].strip().lower(), int(r["fy"]))
        old = out.get(k)
        if old is None:
            out[k] = r
        elif _PRIORITY.get(r.get("origin"), 0) >= _PRIORITY.get(old.get("origin"), 0):
            out[k] = {**{f: old.get(f) for f in FIELDS if old.get(f) is not None}, **{kk: v for kk, v in r.items() if v is not None}}
        else:
            for f in FIELDS:
                if old.get(f) is None and r.get(f) is not None:
                    old[f] = r[f]
    return sorted(out.values(), key=lambda r: (r["company"].lower(), int(r["fy"])))


# ═════════════════════════ 3. EXTRACCIÓN DESDE PDF ═════════════════════════

MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5")
MAX_PAGES = 140
MAX_UPLOAD_MB = 22
KEYWORDS = re.compile(
    r"estado de situaci[oó]n|balance general|estado de resultados|estado del resultado|"
    r"evoluci[oó]n del patrimonio|flujo de efectivo|total del activo|total activo", re.I)

PROMPT = """Sos un analista contable argentino. Del PDF adjunto (estados contables de una concesionaria vial)
extraé los datos para calcular el Altman Z-Score y el Piotroski F-Score. Devolvé SOLO un JSON, sin texto ni markdown, con esta forma:

{"company": "razón social",
 "periods": [
  {"period_end": "YYYY-MM-DD", "column": "current" | "prior",
   "current_assets": n, "current_liabilities": n, "total_assets": n, "total_liabilities": n,
   "equity": n, "retained_earnings": n, "ebit": n, "sales": n,
   "net_income": n, "cfo": n, "long_term_debt": n, "cost_of_sales": n, "share_capital": n,
   "notes": "texto breve"}
 ]}

Reglas:
- Una entrada por cada columna del estado de situación patrimonial: el ejercicio actual ("current") y el comparativo ("prior").
- Números en unidades completas (si el estado dice "en millones", multiplicá por 1.000.000). Pérdidas con signo negativo.
- retained_earnings = reservas (legal, facultativa, otras) + resultados no asignados/acumulados (incluye el resultado del ejercicio). NO incluyas capital social ni ajuste de capital.
- ebit = resultado operativo antes de resultados financieros (intereses, diferencias de cambio, RECPAM, tenencia) y antes de impuesto a las ganancias.
  Si no hay esa línea, calculalo como resultado antes de impuesto menos resultados financieros netos. En "notes" aclará cómo lo obtuviste.
- sales = ingresos operativos (peajes / concesión).
- net_income = resultado neto del ejercicio (después de impuesto a las ganancias).
- cfo = flujo neto de efectivo generado por (usado en) actividades operativas, del estado de flujo de efectivo de cada columna.
- long_term_debt = deudas financieras NO corrientes (préstamos, obligaciones negociables, otras deudas financieras no corrientes). No incluyas proveedores ni deudas fiscales.
- cost_of_sales = costo de los servicios prestados / costo de explotación del ejercicio (en positivo).
- share_capital = capital social nominal (sin ajuste de capital).
- Si un dato no figura, usá null. No inventes valores."""


def select_pages(pdf_bytes: bytes) -> bytes:
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
            **{k: p.get(k) for k in FIELDS},
            "notes": p.get("notes") or "",
        })
    return recs


def extract(pdf_bytes: bytes, api_key: str, source: str, company: str | None = None, model: str = MODEL) -> list[dict]:
    """Envía el PDF a Claude y devuelve registros normalizados.

    La API de Anthropic admite PDFs como bloques de documento codificados en base64.
    Se valida tamaño y se devuelven errores legibles para Streamlit.
    """
    if not api_key:
        raise RuntimeError(
            "No está configurada ANTHROPIC_API_KEY. Configurala en "
            "Streamlit Cloud → Settings → Secrets."
        )

    size_mb = len(pdf_bytes) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_MB:
        raise ValueError(
            f"El PDF pesa {size_mb:.1f} MB. El límite recomendado de esta app es "
            f"{MAX_UPLOAD_MB} MB para enviarlo a Claude."
        )

    try:
        import anthropic
    except ImportError as exc:
        raise RuntimeError(
            "Falta instalar el paquete anthropic. Agregá 'anthropic>=0.75.0' a requirements.txt."
        ) from exc

    selected_pdf = select_pages(pdf_bytes)
    data = base64.standard_b64encode(selected_pdf).decode("utf-8")

    client = anthropic.Anthropic(api_key=api_key)
    try:
        msg = client.messages.create(
            model=model,
            max_tokens=12000,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": data,
                            },
                        },
                        {"type": "text", "text": PROMPT},
                    ],
                }
            ],
        )
    except Exception as exc:
        msg_text = str(exc)
        if "authentication" in msg_text.lower() or "401" in msg_text:
            raise RuntimeError(
                "La ANTHROPIC_API_KEY fue rechazada. Revisá que sea válida y esté activa."
            ) from exc
        if "not_found" in msg_text.lower() or "model" in msg_text.lower() and "404" in msg_text:
            raise RuntimeError(
                f"El modelo '{model}' no está disponible para tu API key. "
                "Podés definir CLAUDE_MODEL en Secrets."
            ) from exc
        if "413" in msg_text or "too large" in msg_text.lower():
            raise RuntimeError(
                "El PDF excede el tamaño máximo aceptado por la API. Probá con un PDF más liviano."
            ) from exc
        raise RuntimeError(f"Error de Anthropic al analizar '{source}': {msg_text}") from exc

    text = "".join(
        b.text for b in msg.content if getattr(b, "type", "") == "text"
    )
    if not text.strip():
        raise RuntimeError("Claude respondió sin contenido de texto.")

    try:
        result = parse_json(text)
    except Exception as exc:
        preview = text[:500].replace("\n", " ")
        raise RuntimeError(
            "Claude no devolvió un JSON válido. Respuesta parcial: " + preview
        ) from exc

    recs = normalize(result, source, company)
    if not recs:
        raise RuntimeError(
            "Claude procesó el PDF pero no encontró ejercicios con fecha válida. "
            "Revisá que el archivo contenga estados contables legibles."
        )
    return recs


# ═════════════════════════ 4. APP STREAMLIT ═════════════════════════
def _secret(name: str) -> str:
    try:
        return str(st.secrets.get(name, "") or "")
    except Exception:  # noqa: BLE001
        return ""


API_KEY = os.getenv("ANTHROPIC_API_KEY", "") or _secret("ANTHROPIC_API_KEY")  # nunca se muestra en pantalla


def api_status() -> tuple[bool, str]:
    if API_KEY:
        return True, "ANTHROPIC_API_KEY configurada"
    return False, "Falta ANTHROPIC_API_KEY"


API_OK, API_STATUS = api_status()

st.set_page_config(page_title="Fundamental Terminal", page_icon="▣", layout="wide")

BLOOMBERG_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

:root {
  --bg: #07090b;
  --panel: #0d1110;
  --panel2: #111714;
  --line: #263229;
  --muted: #7e8c81;
  --text: #e8f0e9;
  --amber: #f5a623;
  --amber2: #ffbf45;
  --green: #3ddc84;
  --red: #ff6262;
}

html, body, [class*="css"] {
  font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
}

.stApp {
  background:
    radial-gradient(900px 480px at 10% -8%, #18221d 0%, #0b0e11 48%, #07090b 100%);
  color: var(--text);
}

header[data-testid="stHeader"] {
  background: #090c0e;
  border-bottom: 1px solid #1c251f;
}

section[data-testid="stSidebar"] {
  background: #080b0d;
  border-right: 1px solid #1d2620;
}

section[data-testid="stSidebar"] * {
  color: #cbd4cc;
}

section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] [data-baseweb="select"] > div,
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
  background: #0f1317 !important;
  border: 1px solid #202a24 !important;
  border-radius: 8px !important;
}

section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
  min-height: 90px;
}

.block-container {
  padding-top: 4.2rem;
  padding-bottom: 3rem;
  max-width: 1480px;
}

h1, h2, h3 {
  font-family: "IBM Plex Sans", sans-serif;
  letter-spacing: .015em;
}

h2 {
  color: #d9e2db;
  font-weight: 600;
}

.ft-masthead {
  display: flex;
  justify-content: space-between;
  align-items: center;
  border: 1px solid #263229;
  background: linear-gradient(90deg, #101612 0%, #0d1210 58%, #17140b 100%);
  padding: 15px 18px 14px;
  margin-bottom: 14px;
  min-height: 118px;
}

.ft-brand {
  font-family: "IBM Plex Mono", monospace;
  color: var(--amber);
  font-weight: 600;
  font-size: 12px;
  letter-spacing: .30em;
  text-transform: uppercase;
}

.ft-title {
  font-size: 27px;
  font-weight: 700;
  color: #eef5ef;
  line-height: 1.08;
  margin-top: 4px;
}

.ft-sub {
  color: #819084;
  font-size: 12px;
  margin-top: 5px;
}

.ft-clock {
  text-align: right;
  font-family: "IBM Plex Mono", monospace;
  color: #9aa79c;
  font-size: 11px;
  line-height: 1.65;
}

.ft-ticker {
  color: var(--amber);
  font-family: "IBM Plex Mono", monospace;
  font-size: 20px;
  font-weight: 600;
}

.ft-company {
  color: #9ca89f;
  font-weight: 600;
  font-size: 14px;
  margin-left: 7px;
}

.ft-cards {
  display: grid;
  grid-template-columns: repeat(6, 1fr);
  gap: 10px;
  margin: 4px 0 22px;
}

.ft-card {
  background: rgba(17,23,20,.94);
  border: 1px solid #263229;
  min-height: 105px;
  padding: 13px 8px 12px;
  text-align: center;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 7px;
}

.ft-clabel {
  font-family: "IBM Plex Mono", monospace;
  color: #89978c;
  letter-spacing: .12em;
  font-size: 10px;
  text-transform: uppercase;
}

.ft-cvalue {
  font-family: "IBM Plex Mono", monospace;
  color: #eef4ef;
  font-size: 1.78rem;
  line-height: 1.12;
}

.ft-cvalue.sm {
  font-size: 1.02rem;
  word-break: break-word;
}

.ft-pill {
  font-size: 11px;
  font-weight: 600;
  border-radius: 999px;
  padding: 2px 10px;
  letter-spacing: .025em;
}

.ft-pill.good {
  background: rgba(61,220,132,.16);
  color: var(--green);
}

.ft-pill.mid {
  background: rgba(245,166,35,.16);
  color: var(--amber2);
}

.ft-pill.bad {
  background: rgba(255,98,98,.15);
  color: var(--red);
}

.stTabs [data-baseweb="tab-list"] {
  gap: 22px;
  background: transparent;
  border-bottom: 1px solid #263229;
}

.stTabs [data-baseweb="tab"] {
  background: transparent;
  color: #e5ece6;
  font-family: "IBM Plex Sans", sans-serif;
  font-weight: 500;
  font-size: 13px;
  padding: 0 0 11px;
}

.stTabs [aria-selected="true"] {
  color: var(--amber) !important;
}

.stTabs [data-baseweb="tab-highlight"] {
  background-color: var(--amber) !important;
  height: 2px;
}

div[data-testid="stDataFrame"] {
  border: 1px solid #263229;
}

[data-testid="stMetric"] {
  background: #111714;
  border: 1px solid #263229;
}

.stButton > button {
  background: var(--amber);
  color: #111;
  border: 0;
  border-radius: 6px;
  font-weight: 700;
  letter-spacing: .07em;
  text-transform: uppercase;
  font-family: "IBM Plex Mono", monospace;
}

.stButton > button:hover {
  background: var(--amber2);
  color: #111;
}

.stDownloadButton > button {
  border-radius: 6px;
}

hr {
  border-color: #263229;
}

section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] * {
  color: #7f8c82;
}

@media (max-width: 1100px) {
  .ft-cards { grid-template-columns: repeat(3, 1fr); }
}

@media (max-width: 700px) {
  .ft-masthead { flex-direction: column; align-items: flex-start; gap: 14px; }
  .ft-clock { text-align: left; }
  .ft-cards { grid-template-columns: repeat(2, 1fr); }
}
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
    if company:
        identity = (
            f'<div style="margin-top:8px">'
            f'<span class="ft-ticker">{escape(company)}</span>'
            f'<span class="ft-company">Concesionaria vial · no cotiza</span>'
            f'</div>'
        )
    else:
        identity = ""

    stmt_html = f"<br/>{escape(stmt)}" if stmt else ""

    st.markdown(
        f"""
        <div class="ft-masthead">
          <div>
            <div class="ft-brand">Fundamental Terminal</div>
            <div class="ft-title">Fundamental Analytics</div>
            <div class="ft-sub">Piotroski F-Score · Altman Z-Score · Concesiones viales</div>
            {identity}
          </div>
          <div class="ft-clock">
            Data: balances contables (PDF)
            <span style="color:#f5a623"> | </span>
            Market: no cotiza
            {stmt_html}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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
if "next_pick" in st.session_state:
    st.session_state["pick_company"] = st.session_state.pop("next_pick")

with st.sidebar:
    st.markdown(
        '<div style="font-family:IBM Plex Mono,monospace;letter-spacing:.12em;'
        'color:#e7eee8;font-size:13px;font-weight:600">COMMAND</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)
    companies = sorted({r["company"] for r in records})
    pick = (st.selectbox("Empresa", companies + ["➕ Nueva empresa…"], key="pick_company")
            if companies else "➕ Nueva empresa…")
    company = st.text_input("Nombre de la empresa (opcional, si no lo detecta del PDF)").strip() if pick.startswith("➕") else pick
    files = st.file_uploader("Balances (PDF)", type="pdf", accept_multiple_files=True)
    go = st.button("ANALIZAR BALANCE", width="stretch", type="primary")
    st.markdown("---")
    st.markdown(
        '<div style="font-family:IBM Plex Mono,monospace;letter-spacing:.10em;'
        'color:#e7eee8;font-size:12px;font-weight:600">MODEL</div>',
        unsafe_allow_html=True,
    )
    model = st.selectbox("Modelo Altman", list(MODELS), index=1,
                         help="Z'' EM suma 3,25 al Z'' y corre los cortes; es la versión para mercados emergentes.")
    ok, status_text = api_status()
    status_bg = "rgba(61,220,132,.12)" if ok else "rgba(255,98,98,.12)"
    status_color = "#3ddc84" if ok else "#ff6262"
    st.markdown(
        f'<div style="margin-top:10px;padding:9px 10px;border:1px solid #263229;'
        f'background:{status_bg};font-family:IBM Plex Mono,monospace;font-size:10px;'
        f'color:{status_color};letter-spacing:.04em">● {status_text}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.caption(f"Motor de extracción: {MODEL}")
    st.caption("Respaldo de datos")
    st.download_button("Descargar JSON", json.dumps(records, ensure_ascii=False, indent=1),
                       "balances.json", "application/json", width="stretch")
    up = st.file_uploader("Restaurar JSON", type="json", key="restore")
    if up is not None and st.button("Restaurar", width="stretch"):
        commit(json.load(up))
        st.rerun()

if go:
    flash = []
    if not files:
        flash.append(("error", "No hay ningún PDF seleccionado."))
    elif not API_KEY:
        flash.append((
            "error",
            "PDF cargado, pero falta ANTHROPIC_API_KEY. Configurala en Streamlit Cloud → Settings → Secrets."
        ))
    else:
        last_company = None
        for f in files:
            pdf_bytes = f.getvalue()
            size_mb = len(pdf_bytes) / (1024 * 1024)
            with st.spinner(f"Analizando {f.name} ({size_mb:.1f} MB)…"):
                try:
                    new = extract(pdf_bytes, API_KEY, f.name, company or None, MODEL)
                    commit(new)
                    last_company = new[0]["company"]
                    flash.append((
                        "success",
                        f"✓ {f.name}: {len(new)} ejercicio(s) detectado(s) para {last_company}."
                    ))
                except Exception as exc:  # noqa: BLE001
                    flash.append(("error", f"✕ {f.name}: {exc}"))
        if last_company:
            st.session_state["next_pick"] = last_company
    st.session_state["flash"] = flash
    st.rerun()

for _kind, _msg in st.session_state.pop("flash", []):
    getattr(st.sidebar, _kind)(_msg)

with st.sidebar:
    st.markdown("---")
    st.markdown("**ABOUT THE MODELS**")
    st.caption("Altman Z'' (1995): Z = 6,56 X1 + 3,26 X2 + 6,72 X3 + 1,05 X4. Cortes 1,10 / 2,60 (EM: +3,25 y 4,15 / 5,85). "
               "X4 usa patrimonio contable porque la empresa no cotiza.")
    st.caption("Piotroski F-Score: 9 criterios binarios (rentabilidad, apalancamiento/liquidez, eficiencia). "
               "8–9 high quality · 4–7 medium · 0–3 low quality. Necesita el ejercicio anterior cargado para los criterios de variación.")
    st.caption("No son calificaciones crediticias. Verificá los datos extraídos en la pestaña DATOS.")

# ───────────────────────── MAIN ─────────────────────────
mine = [r for r in records if company and r["company"].lower() == company.lower()]
if not mine:
    render_masthead(company or None, model)
    if files:
        names = " · ".join(f.name for f in files)
        if API_KEY:
            st.success(f"PDF listo para analizar: {names}")
            st.info("Ahora hacé clic en **ANALIZAR BALANCE**. La app enviará el PDF a Claude y luego mostrará Altman y Piotroski.")
        else:
            st.error("El PDF está cargado, pero falta configurar **ANTHROPIC_API_KEY**.")
            st.markdown(
                "**Streamlit Cloud:** Settings → Secrets → agregá `ANTHROPIC_API_KEY = \"sk-ant-...\"` → Save → Reboot."
            )
    else:
        st.info("Subí uno o más balances en PDF de la concesionaria y luego tocá **ANALIZAR BALANCE**.")
    st.stop()

mine = sorted(mine, key=lambda r: r["fy"])
by_fy = {int(r["fy"]): r for r in mine}
rows, comp, pio = [], [], []
for r in mine:
    c = compute(r, model)
    p = piotroski(r, by_fy.get(int(r["fy"]) - 1))
    rows.append({"FY": str(r["fy"]), "period end": r["period_end"],
                 "Z-Score": None if c["z"] is None else round(c["z"], 2), "zone": c["zone"],
                 "F-Score": p["score"], "criterios evaluados": f"{p['n']}/9", "F zone": p["zone"],
                 **{f"X{i + 1}": None if v is None else round(v, 4) for i, v in enumerate(c["x"])},
                 "origen": r.get("origin", "")})
    comp.append(c)
    pio.append(p)
hist = pd.DataFrame(rows)
last, last_p = comp[-1], pio[-1]
m = MODELS[model]
latest = mine[-1]
render_masthead(company, model, f"Financials: FY{latest['fy']} · period end {latest['period_end']}")

ZK = {"SAFE": "good", "GREY": "mid", "DISTRESS": "bad"}
PK = {"HIGH QUALITY": "good", "MEDIUM": "mid", "LOW QUALITY": "bad"}
ARROW = {"good": "↑", "mid": "→", "bad": "↓", "none": ""}


def cards(items) -> None:
    h = '<div class="ft-cards">'
    for label, value, pill, kind in items:
        sm = " sm" if len(str(value)) > 10 else ""
        p = f'<div class="ft-pill {kind}">{ARROW[kind]} {pill}</div>' if pill else ""
        h += f'<div class="ft-card"><div class="ft-clabel">{label}</div><div class="ft-cvalue{sm}">{escape(str(value))}</div>{p}</div>'
    st.markdown(h + "</div>", unsafe_allow_html=True)


def pio_table(p: dict) -> pd.DataFrame:
    return pd.DataFrame({
        "signal": [x[0] for x in p["spec"]], "name": [x[1] for x in p["spec"]],
        "points": pd.array([None if x[2] is None else int(bool(x[2])) for x in p["spec"]], dtype="Int64"),
        "detail": [x[3] for x in p["spec"]]})


def alt_table(c: dict) -> pd.DataFrame:
    return pd.DataFrame({
        "factor": FACTOR_NAMES, "peso": m["w"],
        "ratio": [None if v is None else round(v, 4) for v in c["x"]],
        "contribución": [None if v is None or not w else round(v, 4) for v, w in zip(c["contrib"], m["w"])]})


zval = "—" if last["z"] is None else f"{last['z']:.2f}"
fval = "—" if last_p["score"] is None else f"{last_p['score']}/9"
cards([
    ("COMPANY", company, "", "none"),
    ("F-SCORE", fval, last_p["zone"] if last_p["score"] is not None else "", PK.get(last_p["zone"], "none")),
    ("Z-SCORE", zval, last["zone"] if last["z"] is not None else "", ZK.get(last["zone"], "none")),
    ("TOTAL ASSETS", _fmt_num(latest.get("total_assets")), "", "none"),
    ("BOOK EQUITY", _fmt_num(latest.get("equity")), "", "none"),
    ("FY", str(latest["fy"]), "", "none"),
])

st.markdown('<div style="height:2px"></div>', unsafe_allow_html=True)
tab_o, tab_p, tab_c, tab_h, tab_f, tab_g, tab_d = st.tabs(
    ["OVERVIEW", "PIOTROSKI", "ALTMAN", "HISTORIAL", "FORMULAS", "CHARTS", "DATOS"])

with tab_o:
    st.header("Piotroski F-Score")
    st.dataframe(pio_table(last_p), width="stretch", hide_index=True)
    st.markdown(f"**F-Score = {fval}** · {last_p['zone']} · criterios evaluados: {last_p['n']}/9")
    if last_p["n"] < 9:
        st.caption("Faltan criterios: cargá también el balance del ejercicio anterior o completá los campos en DATOS.")
    st.header("Altman Z-Score")
    st.dataframe(alt_table(last), width="stretch", hide_index=True)
    st.markdown(f"Constante: `{m['const']}` · **Z = {'n/a' if last['z'] is None else format(last['z'], '.3f')}** · {last['zone']}")
    if last["missing"]:
        st.warning("Faltan datos para: " + ", ".join(last["missing"]))

with tab_p:
    yr2 = st.selectbox("Ejercicio", list(hist["FY"])[::-1], key="yr_pio")
    p = pio[list(hist["FY"]).index(yr2)]
    st.dataframe(pio_table(p), width="stretch", hide_index=True)
    st.markdown(f"**F-Score = {p['score'] if p['score'] is not None else 'n/a'} / 9** · {p['zone']} · criterios evaluados: {p['n']}/9")
    if p["n"] < 9:
        st.warning("Hay criterios sin dato. Cargá el balance del ejercicio anterior o completá en DATOS: "
                   "net_income, cfo, long_term_debt, cost_of_sales, share_capital.")

with tab_c:
    yr = st.selectbox("Ejercicio", list(hist["FY"])[::-1], key="yr_altman")
    c = comp[list(hist["FY"]).index(yr)]
    st.dataframe(alt_table(c), width="stretch", hide_index=True)
    ztxt = "n/a" if c["z"] is None else format(c["z"], ".3f")
    st.markdown(f"Constante: `{m['const']}` · **Z = {ztxt}** · {c['zone']}")
    if c["missing"]:
        st.warning("Faltan datos para: " + ", ".join(c["missing"]))
    notes = [r.get("notes") for r in mine if str(r["fy"]) == yr and r.get("notes")]
    if notes:
        st.caption("Notas de extracción: " + " | ".join(notes))

with tab_h:
    st.dataframe(hist, width="stretch", hide_index=True)
    zs = [z for z in hist["Z-Score"] if z is not None and not pd.isna(z)]
    if zs:
        st.markdown(f"**Altman promedio:** `{sum(zs) / len(zs):.2f}` · Mín `{min(zs):.2f}` · Máx `{max(zs):.2f}`")
    st.caption("Los ratios usan valores de una misma columna del balance, por lo que la reexpresión por inflación no los distorsiona.")

with tab_g:
    st.subheader("Evolución de los indicadores")
    st.caption("Ejercicios cargados, del más antiguo al más reciente.")
    d = hist.dropna(subset=["Z-Score"])
    if d.empty:
        st.info("No hay datos suficientes para graficar Altman.")
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
        st.caption("Barras: peso × ratio de cada factor. El rombo ámbar es el Z-Score total.")

    dp = hist.dropna(subset=["F-Score"])
    if not dp.empty:
        yrs = list(dp["FY"])
        fb = alt.Chart(dp).encode(
            x=alt.X("FY:N", title="Fiscal year", sort=yrs, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("F-Score:Q", title="F-Score", scale=alt.Scale(domain=[0, 9])),
            tooltip=["FY", "F-Score", "criterios evaluados", "F zone"])
        fbars = fb.mark_bar(color="#3ddc84", opacity=0.85)
        flbl = fb.mark_text(dy=-8, color="#e8f3e9", fontSize=12, font="IBM Plex Mono").encode(text="F-Score:Q")
        fz = _threshold_layer([(8, "FUERTE ≥ 8", "#3ddc84"), (3.5, "DÉBIL ≤ 3", "#ff5c5c")])
        st.altair_chart(_style_chart((fz + fbars + flbl).properties(height=300, title="Piotroski F-Score")), width="stretch")

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
**Altman** — **X1** = (Act. corriente − Pas. corriente) / Activo · **X2** = Resultados acumulados / Activo ·
**X3** = EBIT / Activo · **X4** = **Patrimonio neto contable** / Pasivo · **X5** = Ventas / Activo

| Modelo | Fórmula | Cortes (distress / safe) |
|---|---|---|
| Z'' | 6,56·X1 + 3,26·X2 + 6,72·X3 + 1,05·X4 | 1,10 / 2,60 |
| Z'' EM | 3,25 + Z'' | 4,15 / 5,85 |
| Z' | 0,717·X1 + 0,847·X2 + 3,107·X3 + 0,420·X4 + 0,998·X5 | 1,23 / 2,90 |

**Piotroski** — 1 punto por cada criterio cumplido:
1. Resultado neto > 0 · 2. CFO > 0 · 3. ROA (RN/Activo) mayor que el año anterior · 4. CFO > Resultado neto ·
5. Deuda financiera no corriente / Activo menor que el año anterior · 6. Liquidez corriente mayor ·
7. Capital social nominal sin aumento · 8. Margen bruto ((Ventas − Costo)/Ventas) mayor · 9. Rotación (Ventas/Activo) mayor.
Puntaje 8–9 fuerte, 4–7 medio, 0–3 débil. Para una no cotizante, el criterio 7 se mide con el capital social nominal.
""")










