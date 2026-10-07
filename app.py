"""Fundamental Terminal — Streamlit desk for SEC XBRL fundamental scores."""

from __future__ import annotations

import datetime as dt
import json
import math
import re
import sys
import urllib.request
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import altair as alt
import pandas as pd
import streamlit as st

from src.altman import compute_altman
from src.piotroski import compute_piotroski
from src.sec_loader import (
    CompanyMeta,
    extract_statement_bundle,
    fetch_market_snapshot,
    load_company_facts,
    load_submissions,
    enrich_meta,
    resolve_ticker,
    SecLoaderError,
)

st.set_page_config(
    page_title="Fundamental Terminal",
    page_icon="▣",
    layout="wide",
    initial_sidebar_state="expanded",
)

BLOOMBERG_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

html, body, [class*="css"]  {
    font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
}

.stApp {
    background: radial-gradient(1200px 600px at 10% -10%, #1a2420 0%, #0b0e11 45%, #07090b 100%);
    color: #d7e0d8;
}

header[data-testid="stHeader"] {
    background: #0b0e11;
    border-bottom: 1px solid #1f2a22;
}

section[data-testid="stSidebar"] {
    background: #0a0d10;
    border-right: 1px solid #1f2a22;
}

section[data-testid="stSidebar"] * {
    color: #c5d0c6;
}

.block-container {
    padding-top: 4.5rem;
    max-width: 1480px;
}

h1, h2, h3 {
    font-family: "IBM Plex Sans", sans-serif;
    letter-spacing: 0.04em;
}

.ft-masthead {
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    border: 1px solid #243328;
    background: linear-gradient(90deg, #101612 0%, #0d1210 60%, #151208 100%);
    padding: 14px 18px 12px 18px;
    margin-bottom: 14px;
}
.ft-brand {
    font-family: "IBM Plex Mono", monospace;
    color: #f5a623;
    font-weight: 600;
    font-size: 13px;
    letter-spacing: 0.28em;
    text-transform: uppercase;
    line-height: 1.4;
}
.ft-title {
    font-size: 28px;
    font-weight: 700;
    color: #eef6ef;
    line-height: 1.1;
    margin-top: 4px;
}
.ft-sub {
    color: #7f8f82;
    font-size: 13px;
    margin-top: 4px;
}
.ft-clock {
    text-align: right;
    font-family: "IBM Plex Mono", monospace;
    color: #9aa89b;
    font-size: 12px;
}
.ft-ticker {
    color: #f5a623;
    font-size: 22px;
    font-weight: 600;
}

div[data-testid="stMetric"] {
    background: #111714;
    border: 1px solid #243328;
    padding: 12px 8px;
    text-align: center;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}
div[data-testid="stMetric"] [data-testid="stMetricLabel"],
div[data-testid="stMetric"] [data-testid="stMetricValue"],
div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
    width: 100%;
    justify-content: center;
    text-align: center;
}
div[data-testid="stMetric"] [data-testid="stMetricLabel"] *,
div[data-testid="stMetric"] [data-testid="stMetricValue"] *,
div[data-testid="stMetric"] [data-testid="stMetricDelta"] * {
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: clip !important;
    text-align: center;
    justify-content: center;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] > div {
    font-size: 1.7rem;
    line-height: 1.2;
}
div[data-testid="stMetric"] label {
    color: #8b9a8d !important;
    font-family: "IBM Plex Mono", monospace;
    letter-spacing: 0.12em;
    font-size: 11px !important;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-family: "IBM Plex Mono", monospace;
    color: #e8f3e9;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: #0d1110;
    border-bottom: 1px solid #243328;
}
.stTabs [data-baseweb="tab"] {
    background: #0d1110;
    color: #8b9a8d;
    font-family: "IBM Plex Mono", monospace;
    letter-spacing: 0.08em;
}
.stTabs [aria-selected="true"] {
    color: #f5a623 !important;
    border-bottom: 2px solid #f5a623;
}

.pass { color: #3ddc84; font-weight: 600; }
.fail { color: #ff5c5c; font-weight: 600; }
.miss { color: #f5a623; font-weight: 600; }

.zone-SAFE { color: #3ddc84; }
.zone-GREY { color: #f5a623; }
.zone-DISTRESS { color: #ff5c5c; }

.stButton>button {
    background: #f5a623;
    color: #111;
    border: 0;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-family: "IBM Plex Mono", monospace;
}
.stButton>button:hover {
    background: #ffc056;
    color: #111;
}

hr { border-color: #243328; }

.stDataFrame { border: 1px solid #243328; }

.audit-note {
    font-family: "IBM Plex Mono", monospace;
    font-size: 12px;
    color: #8b9a8d;
}
</style>
"""

st.markdown(BLOOMBERG_CSS, unsafe_allow_html=True)


def _fmt_num(v, kind: str = "int") -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "—"
    if kind == "pct":
        return f"{v:.2%}"
    if kind == "ratio":
        return f"{v:.4f}"
    if kind == "score":
        return f"{v:.2f}"
    if abs(v) >= 1_000_000_000_000:
        return f"{v/1_000_000_000_000:,.2f}T"
    if abs(v) >= 1_000_000_000:
        return f"{v/1_000_000_000:,.2f}B"
    if abs(v) >= 1_000_000:
        return f"{v/1_000_000:,.1f}M"
    if abs(v) >= 10_000:
        return f"{v:,.0f}"
    return f"{v:,.2f}"


def _fmt_usd(v, compact: bool = False) -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "—"
    return "$" + (_fmt_num(v) if compact else f"{v:,.2f}")


def _pass_label(passed: bool | None) -> str:
    if passed is True:
        return "PASS · 1"
    if passed is False:
        return "FAIL · 0"
    return "N/A · —"


_BIG_NUM = re.compile(r"-?\d{1,3}(?:,\d{3}){2,}(?:\.\d+)?")
_PCT_NUM = re.compile(r"(-?\d+\.\d+)%")
_DEC_NUM = re.compile(r"(?<![\d.,])(-?\d+\.\d{5,})(?![\d%])")


def _pretty_detail(text) -> str:
    """Shorten big numbers (111,482,000,000 -> 111.48B) and trim long decimals."""
    if text is None:
        return ""
    t = str(text)
    t = _BIG_NUM.sub(lambda m: _fmt_num(float(m.group(0).replace(",", ""))), t)
    t = _PCT_NUM.sub(lambda m: f"{float(m.group(1)):.2f}%", t)
    t = _DEC_NUM.sub(lambda m: f"{float(m.group(1)):.4f}", t)
    return t


@st.cache_data(show_spinner=False, ttl=60 * 30)
def cached_resolve(ticker: str) -> CompanyMeta:
    return resolve_ticker(ticker)


@st.cache_data(show_spinner=False, ttl=60 * 30)
def cached_facts(cik: str) -> dict:
    return load_company_facts(cik)


@st.cache_data(show_spinner=False, ttl=60 * 30)
def cached_subs(cik: str) -> dict:
    return load_submissions(cik)


@st.cache_data(show_spinner=False, ttl=60 * 60 * 12)
def cached_close_on(ticker: str, date_str: str) -> dict:
    """Last daily close on or before date_str (Yahoo chart API, split-adjusted)."""
    try:
        d = dt.date.fromisoformat(str(date_str)[:10])
        utc = dt.timezone.utc
        p1 = int(dt.datetime.combine(d - dt.timedelta(days=10), dt.time.min, tzinfo=utc).timestamp())
        p2 = int(dt.datetime.combine(d + dt.timedelta(days=2), dt.time.min, tzinfo=utc).timestamp())
        url = (
            f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
            f"?period1={p1}&period2={p2}&interval=1d&events=history"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.load(resp)
        return _parse_close_on(data, d)
    except Exception as exc:  # noqa: BLE001
        return {"price": None, "date": None, "error": str(exc)}


def _parse_close_on(data: dict, target: dt.date) -> dict:
    result = data["chart"]["result"][0]
    stamps = result.get("timestamp") or []
    closes = result["indicators"]["quote"][0].get("close") or []
    offset = (result.get("meta") or {}).get("gmtoffset", 0) or 0
    best = None
    for t, c in zip(stamps, closes):
        if c is None:
            continue
        day = (dt.datetime.fromtimestamp(t, dt.timezone.utc) + dt.timedelta(seconds=offset)).date()
        if day <= target and (best is None or day > best[0]):
            best = (day, float(c))
    if best is None:
        return {"price": None, "date": None, "error": "no close on or before target date"}
    return {"price": best[1], "date": best[0].isoformat(), "error": None}


@st.cache_data(show_spinner=False, ttl=60 * 15)
def cached_mkt(ticker: str) -> dict:
    return fetch_market_snapshot(ticker)


def _statement_point(bundle: dict, year):
    """Latest-period data point of the fiscal year used for the scores."""
    if year is None:
        return None
    best = None
    for pts in bundle.values():
        for p in pts:
            if p.fy == year and p.end and (best is None or str(p.end) > str(best.end)):
                best = p
    return best


def _statement_info(bundle: dict, year) -> str:
    """Describe the filing behind the scores: fiscal year, period end date and form."""
    if year is None:
        return ""
    best = _statement_point(bundle, year)
    if best is None:
        return f"Financials: FY{year}"
    parts = [f"Financials: FY{year}", f"period end {best.end}"]
    if best.form:
        parts.append(str(best.form))
    return " · ".join(parts)


def _trunc_bundle(bundle: dict, year: int) -> dict:
    """Bundle restricted to fiscal years <= year, so the models treat `year` as the latest."""
    return {k: [p for p in pts if p.fy is not None and p.fy <= year] for k, pts in bundle.items()}


def build_history(bundle: dict, ticker: str, n: int = 5, latest_cap=None, latest_price=None):
    """Piotroski and Altman for each of the last n fiscal years (newest first)."""
    years = sorted({p.fy for pts in bundle.values() for p in pts if p.fy is not None})[-n:]
    latest = years[-1] if years else None
    f_rows, z_rows = [], []
    for y in reversed(years):
        tb = _trunc_bundle(bundle, y)
        pt = _statement_point(tb, y)
        end = str(pt.end)[:10] if pt is not None else None

        # Piotroski
        try:
            fr = compute_piotroski(tb)
        except Exception:  # noqa: BLE001
            fr = None
        row = {"FY": str(y), "period end": end, "F-Score": None, "quality": "n/a"}
        if fr is not None and fr.year == y:
            row["F-Score"] = fr.score
            row["quality"] = fr.label
            for cr in fr.criteria:
                row[cr.code] = cr.points if cr.passed is not None else None
        f_rows.append(row)

        # Altman: close price on the statement date of that year
        if y == latest and (latest_cap or latest_price):
            cap, price = latest_cap, latest_price
        else:
            hp = cached_close_on(ticker, end) if end else {"price": None}
            cap, price = None, hp.get("price")
        zrow = {"FY": str(y), "period end": end, "close (USD)": price, "mkt cap": "—",
                "Z-Score": None, "zone": "n/a"}
        try:
            zr = compute_altman(tb, market_cap=cap, market_price=price)
        except Exception:  # noqa: BLE001
            zr = None
        if zr is not None and zr.year == y and zr.z is not None:
            zrow["mkt cap"] = _fmt_usd(zr.market_cap, compact=True)
            zrow["Z-Score"] = round(zr.z, 2)
            zrow["zone"] = zr.zone
            for cm in zr.components:
                zrow[cm.code] = None if cm.value is None else round(cm.value, 4)
        z_rows.append(zrow)
    return f_rows, z_rows


def _avg(vals):
    v = [x for x in vals if x is not None]
    return (sum(v) / len(v)) if v else None


# ───────────────────────────── CHARTS ─────────────────────────────

def _style_chart(ch: alt.Chart) -> alt.Chart:
    """Dark terminal look for Altair charts."""
    return (
        ch.configure(background="#0b0e11")
        .configure_view(stroke="#243328")
        .configure_axis(
            labelColor="#8b9a8d", titleColor="#8b9a8d",
            gridColor="#1f2a22", domainColor="#243328", tickColor="#243328",
            labelFont="IBM Plex Mono", titleFont="IBM Plex Mono",
        )
        .configure_legend(labelColor="#8b9a8d", titleColor="#8b9a8d")
        .configure_title(color="#e8f3e9", font="IBM Plex Mono", fontSize=14)
    )


def _threshold_layer(levels: list[tuple[float, str, str]]) -> alt.Chart:
    """Dashed horizontal reference lines: (value, label, color)."""
    df = pd.DataFrame(levels, columns=["y", "label", "color"])
    rules = alt.Chart(df).mark_rule(strokeDash=[5, 4], opacity=0.8).encode(
        y="y:Q", color=alt.Color("color:N", scale=None)
    )
    texts = alt.Chart(df).mark_text(align="left", dx=4, dy=-6, fontSize=10).encode(
        y="y:Q", text="label:N", color=alt.Color("color:N", scale=None),
        x=alt.value(4),
    )
    return rules + texts


def piotroski_chart(hist_f: list[dict]) -> alt.Chart | None:
    df = pd.DataFrame(hist_f)
    if df.empty or "F-Score" not in df:
        return None
    df = df.dropna(subset=["F-Score"]).sort_values("FY")
    if df.empty:
        return None
    base = alt.Chart(df).encode(
        x=alt.X("FY:N", title="Fiscal year", sort=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("F-Score:Q", title="F-Score", scale=alt.Scale(domain=[0, 9])),
        tooltip=["FY", "period end", "F-Score", "quality"],
    )
    line = base.mark_line(color="#f5a623", strokeWidth=2.5,
                          point=alt.OverlayMarkDef(color="#f5a623", size=90, filled=True))
    labels = base.mark_text(dy=-14, color="#e8f3e9", fontSize=12,
                            font="IBM Plex Mono").encode(text="F-Score:Q")
    zones = _threshold_layer([
        (8, "HIGH ≥ 8", "#3ddc84"),
        (5, "AVERAGE ≥ 5", "#8b9a8d"),
    ])
    return _style_chart((zones + line + labels).properties(height=340, title="Piotroski F-Score"))


def altman_chart(hist_z: list[dict]) -> alt.Chart | None:
    df = pd.DataFrame(hist_z)
    if df.empty or "Z-Score" not in df:
        return None
    df = df.dropna(subset=["Z-Score"]).sort_values("FY")
    if df.empty:
        return None
    lo = min(0.0, float(df["Z-Score"].min()) - 0.5)
    hi = max(3.5, float(df["Z-Score"].max()) + 0.5)
    base = alt.Chart(df).encode(
        x=alt.X("FY:N", title="Fiscal year", sort=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Z-Score:Q", title="Z-Score", scale=alt.Scale(domain=[lo, hi])),
        tooltip=["FY", "period end", "Z-Score", "zone", "close (USD)"],
    )
    line = base.mark_line(color="#f5a623", strokeWidth=2.5,
                          point=alt.OverlayMarkDef(color="#f5a623", size=90, filled=True))
    labels = base.mark_text(dy=-14, color="#e8f3e9", fontSize=12,
                            font="IBM Plex Mono").encode(text=alt.Text("Z-Score:Q", format=".2f"))
    zones = _threshold_layer([
        (2.99, "SAFE > 2.99", "#3ddc84"),
        (1.81, "DISTRESS < 1.81", "#ff5c5c"),
    ])
    return _style_chart((zones + line + labels).properties(height=340, title="Altman Z-Score"))


def piotroski_criteria_chart(hist_f: list[dict]) -> alt.Chart | None:
    """Barras apiladas: cada señal aprobada suma 1 punto a la barra del año."""
    df = pd.DataFrame(hist_f)
    if df.empty:
        return None
    meta_cols = {"FY", "period end", "F-Score", "quality"}
    codes = [c for c in df.columns if c not in meta_cols]
    if not codes:
        return None
    long = (
        df.melt(id_vars=["FY", "F-Score"], value_vars=codes,
                var_name="signal", value_name="points")
        .dropna(subset=["points"])
    )
    long = long[long["points"] > 0]
    if long.empty:
        return None
    years = sorted(df["FY"].unique())

    bars = alt.Chart(long).mark_bar(stroke="#0b0e11", strokeWidth=1).encode(
        x=alt.X("FY:N", title="Fiscal year", sort=years, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("sum(points):Q", title="Señales aprobadas",
                scale=alt.Scale(domain=[0, 9])),
        color=alt.Color("signal:N", title="Señal", sort=codes,
                        scale=alt.Scale(scheme="tableau10")),
        order=alt.Order("signal:N", sort="ascending"),
        tooltip=["FY", "signal", "points"],
    )
    totals = (
        alt.Chart(df.dropna(subset=["F-Score"]))
        .mark_text(dy=-8, color="#e8f3e9", fontSize=13, font="IBM Plex Mono")
        .encode(
            x=alt.X("FY:N", sort=years),
            y=alt.Y("F-Score:Q", scale=alt.Scale(domain=[0, 9])),
            text=alt.Text("F-Score:Q", format="d"),
        )
    )
    return _style_chart((bars + totals).properties(height=360, title="Piotroski: criterios por año"))


_ALTMAN_WEIGHTS = [1.2, 1.4, 3.3, 0.6, 1.0]


def altman_contrib_chart(hist_z: list[dict]) -> alt.Chart | None:
    """Barras apiladas con la contribución ponderada de X1–X5 y el Z total encima."""
    df = pd.DataFrame(hist_z)
    if df.empty or "Z-Score" not in df:
        return None
    meta_cols = {"FY", "period end", "close (USD)", "mkt cap", "Z-Score", "zone"}
    factors = [c for c in df.columns if c not in meta_cols][: len(_ALTMAN_WEIGHTS)]
    if not factors:
        return None
    df = df.dropna(subset=["Z-Score"]).copy()
    if df.empty:
        return None
    for col, w in zip(factors, _ALTMAN_WEIGHTS):
        df[col] = pd.to_numeric(df[col], errors="coerce") * w   # ratio -> contribución
    long = df.melt(id_vars=["FY", "Z-Score"], value_vars=factors,
                   var_name="factor", value_name="contribution").dropna(subset=["contribution"])
    years = sorted(df["FY"].unique())

    bars = alt.Chart(long).mark_bar(stroke="#0b0e11", strokeWidth=1).encode(
        x=alt.X("FY:N", title="Fiscal year", sort=years, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("contribution:Q", title="Contribución al Z-Score"),
        color=alt.Color("factor:N", title="Factor", sort=factors,
                        scale=alt.Scale(scheme="tableau10")),
        order=alt.Order("factor:N", sort="ascending"),
        tooltip=["FY", "factor", alt.Tooltip("contribution:Q", format=".3f")],
    )
    total = (
        alt.Chart(df)
        .mark_point(shape="diamond", size=140, filled=True, color="#f5a623",
                    stroke="#0b0e11")
        .encode(x=alt.X("FY:N", sort=years), y="Z-Score:Q",
                tooltip=["FY", alt.Tooltip("Z-Score:Q", format=".2f"), "zone"])
    )
    total_lbl = (
        alt.Chart(df)
        .mark_text(dx=22, color="#f5a623", fontSize=12, font="IBM Plex Mono")
        .encode(x=alt.X("FY:N", sort=years), y="Z-Score:Q",
                text=alt.Text("Z-Score:Q", format=".2f"))
    )
    zones = _threshold_layer([
        (2.99, "SAFE > 2.99", "#3ddc84"),
        (1.81, "DISTRESS < 1.81", "#ff5c5c"),
    ])
    return _style_chart((zones + bars + total + total_lbl).properties(
        height=380, title="Altman: contribución de X1–X5 por año"))


# ───────────────────────────── UI ─────────────────────────────

def render_masthead(meta: CompanyMeta | None, mkt: dict | None, stmt: str = "") -> None:
    ticker_html = ""
    stmt_html = f"<br/>{stmt}" if stmt else ""
    if meta:
        ticker_html = f'<div class="ft-ticker">{meta.ticker} &nbsp;<span style="color:#9aa89b;font-size:14px">{meta.name}</span></div>'
    st.markdown(
        f"""
        <div class="ft-masthead">
          <div>
            <div class="ft-brand">Fundamental Terminal</div>
            <div class="ft-title">SEC Fundamental Analytics</div>
            <div class="ft-sub">Piotroski F-Score · Altman Z-Score</div>
            {ticker_html}
          </div>
          <div class="ft-clock">Data: data.sec.gov &nbsp;|&nbsp; Market: Yahoo (optional){stmt_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


with st.sidebar:
    st.markdown("**COMMAND**")
    ticker = st.text_input("Ticker", value="AAPL", placeholder="e.g. AAPL, MSFT, JPM").strip().upper()
    run = st.button("Load SEC XBRL", width="stretch")
    st.markdown("---")
    st.markdown("**ABOUT THE MODELS**")
    st.caption(
        "Piotroski (2000): nine binary fundamental signals on profitability, "
        "leverage/liquidity/dilution and operating efficiency. Score 0–9."
    )
    st.caption(
        "Altman (1968) original manufacturing Z: "
        "1.2 X1 + 1.4 X2 + 3.3 X3 + 0.6 X4 + 1.0 X5. "
        "Cutoffs 1.81 / 2.99. Not a credit rating."
    )
    st.markdown("---")
    st.caption("Educational desk. Filings can use heterogeneous US-GAAP concepts.")

if "loaded_ticker" not in st.session_state:
    st.session_state.loaded_ticker = ticker or "AAPL"

if run:
    st.session_state.loaded_ticker = ticker

active = st.session_state.loaded_ticker

if not active:
    render_masthead(None, None)
    st.info("Enter a ticker in the sidebar and press **Load SEC XBRL**.")
    st.stop()

try:
    with st.spinner(f"Resolving {active} on SEC EDGAR…"):
        meta = replace(cached_resolve(active))
        subs = cached_subs(meta.cik)
        meta = enrich_meta(meta, subs)
        facts = cached_facts(meta.cik)
        bundle = extract_statement_bundle(facts)
        mkt = cached_mkt(active)
except SecLoaderError as exc:
    st.error(str(exc))
    st.stop()
except Exception as exc:  # noqa: BLE001
    st.error(f"Failed to load SEC data: {exc}")
    st.stop()

f_res = compute_piotroski(bundle)

# X4 uses the closing price on the statement date (last trading day on or before it).
_pt = _statement_point(bundle, f_res.year)
stmt_end = str(_pt.end)[:10] if _pt is not None else None
hist = cached_close_on(active, stmt_end) if stmt_end else {"price": None, "date": None, "error": "no statement date"}

x4_cap = None
x4_price = None
if hist.get("price"):
    x4_price = float(hist["price"])
    x4_note = (
        f"X4 price: close {x4_price:,.2f} on {hist['date']} "
        f"(last trading day on or before statement date {stmt_end})."
    )
else:
    x4_cap = mkt.get("market_cap") if mkt else None
    x4_price = mkt.get("price") if mkt else None
    x4_note = (
        f"Could not get the close for {stmt_end or 'the statement date'}"
        f"{' (' + str(hist.get('error')) + ')' if hist.get('error') else ''}; "
        "X4 uses current market data instead."
    )

z_res = compute_altman(bundle, market_cap=x4_cap, market_price=x4_price)

render_masthead(meta, mkt, _statement_info(bundle, f_res.year or z_res.year))

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("TICKER", meta.ticker)
c2.metric("F-SCORE", "—" if f_res.score is None else f"{f_res.score}/9", f_res.label)
c3.metric("Z-SCORE", "—" if z_res.z is None else f"{z_res.z:.2f}", z_res.zone)
_px_shown = x4_price if x4_price else (mkt.get("price") if mkt else None)
_cap_shown = z_res.market_cap if z_res.market_cap is not None else (mkt.get("market_cap") if mkt else None)
c4.metric("PRICE @ FY END (USD)", _fmt_usd(_px_shown))
c5.metric("MKT CAP @ FY END (USD)", _fmt_usd(_cap_shown, compact=True))
c6.metric("FY", str(f_res.year or z_res.year or "—"))

with st.spinner("Building 5-year history…"):
    hist_f, hist_z = build_history(bundle, active, 5, latest_cap=x4_cap, latest_price=x4_price)

tab_ov, tab_f, tab_z, tab_form, tab_charts = st.tabs(
    ["OVERVIEW", "PIOTROSKI", "ALTMAN", "FORMULAS", "CHARTS"]
)

with tab_ov:
    st.subheader("Piotroski F-Score")
    rows = []
    for cr in f_res.criteria:
        rows.append(
            {
                "signal": cr.code,
                "name": cr.name,
                "points": cr.points if cr.passed is not None else None,
                "detail": _pretty_detail(cr.narrative),
            }
        )
    st.dataframe(
        pd.DataFrame(rows),
        width="stretch",
        hide_index=True,
        column_config={
            "signal": st.column_config.TextColumn("signal", width="small"),
            "name": st.column_config.TextColumn("name", width="medium"),
            "points": st.column_config.NumberColumn("points", width="small", format="%d"),
            "detail": st.column_config.TextColumn("detail", width="large"),
        },
    )
    st.markdown(
        f"**Score:** `{'n/a' if f_res.score is None else f'{f_res.score}/9'}` — {f_res.label}  \n"
        f"Fiscal years: {f_res.prior_year} → {f_res.year}"
    )
    if f_res.missing:
        st.warning("Incomplete signals (counted as 0): " + ", ".join(f_res.missing))

    st.markdown("**Historical Piotroski (last 5 fiscal years)**")
    st.dataframe(pd.DataFrame(hist_f), width="stretch", hide_index=True)
    _fs = [r["F-Score"] for r in hist_f]
    _fv = [x for x in _fs if x is not None]
    if _fv:
        st.markdown(
            f"**Average F-Score:** `{_avg(_fv):.2f}/9` over {len(_fv)} fiscal years  \n"
            f"Min `{min(_fv)}` · Max `{max(_fv)}`"
        )
    else:
        st.caption("Not enough data to build the Piotroski history.")

    st.subheader("Altman Z-Score")
    zrows = []
    for cm in z_res.components:
        zrows.append(
            {
                "factor": cm.code,
                "name": cm.name,
                "weight": cm.weight,
                "ratio": None if cm.value is None else round(cm.value, 4),
                "contribution": None if cm.contribution is None else round(cm.contribution, 4),
            }
        )
    st.dataframe(
        pd.DataFrame(zrows),
        width="stretch",
        hide_index=True,
        column_config={
            "weight": st.column_config.NumberColumn("weight", format="%.1f"),
            "ratio": st.column_config.NumberColumn("ratio", format="%.4f"),
            "contribution": st.column_config.NumberColumn("contribution", format="%.4f"),
        },
    )
    st.markdown(
        f"**Z-Score:** `{'n/a' if z_res.z is None else f'{z_res.z:.2f}'}`\n\n"
        f"**Zone:** `{z_res.zone}` — {z_res.zone_detail}  \n"
        f"Market equity source: {z_res.market_cap_source or 'unavailable'}  \n"
        f"{x4_note}"
    )
    if z_res.missing:
        st.warning("Missing Altman factors: " + ", ".join(z_res.missing))

    st.markdown("**Historical Altman (last 5 fiscal years)**")
    st.caption("X4 uses the close on each statement date × shares outstanding.")
    st.dataframe(
        pd.DataFrame(hist_z),
        width="stretch",
        hide_index=True,
        column_config={"close (USD)": st.column_config.NumberColumn("close (USD)", format="$%.2f")},
    )
    _zv = [r["Z-Score"] for r in hist_z if r["Z-Score"] is not None]
    if _zv:
        st.markdown(
            f"**Average Z-Score:** `{_avg(_zv):.2f}` over {len(_zv)} fiscal years  \n"
            f"Min `{min(_zv):.2f}` · Max `{max(_zv):.2f}`"
        )
    else:
        st.caption("Not enough data to build the Altman history.")

with tab_f:
    st.markdown(
        f"Fiscal years **{f_res.prior_year} → {f_res.year}**. "
        f"Score **{f_res.score if f_res.score is not None else 'n/a'} / 9** · {f_res.label}."
    )
    for cr in f_res.criteria:
        color = "pass" if cr.passed else ("fail" if cr.passed is False else "miss")
        with st.expander(f"{cr.code}  ·  {cr.name}  ·  {_pass_label(cr.passed)}", expanded=False):
            st.markdown(f"<span class='{color}'>{_pass_label(cr.passed)}</span> — {cr.explanation}", unsafe_allow_html=True)
            st.latex(cr.latex)
            st.code(cr.formula, language="text")
            st.markdown(cr.narrative)
            pretty = {k: v for k, v in cr.inputs.items() if not str(k).endswith((".accn",))}
            st.json({k: (round(v, 6) if isinstance(v, float) else v) for k, v in pretty.items()})

with tab_z:
    st.markdown(
        r"$Z = 1.2X_1 + 1.4X_2 + 3.3X_3 + 0.6X_4 + 1.0X_5$"
        + (f"  =  **{z_res.z:.3f}**" if z_res.z is not None else "  =  *incomplete*")
    )
    st.caption(z_res.zone_detail)
    for cm in z_res.components:
        with st.expander(
            f"{cm.code}  ·  {cm.name}  ·  "
            f"{'—' if cm.contribution is None else f'{cm.weight} × {cm.value:.4f} = {cm.contribution:.4f}'}",
            expanded=False,
        ):
            st.latex(cm.latex)
            st.code(cm.formula, language="text")
            st.write(cm.explanation)
            pretty = {k: v for k, v in cm.inputs.items()}
            st.json({k: (round(v, 6) if isinstance(v, float) else v) for k, v in pretty.items()})

with tab_form:
    st.subheader("Piotroski F-Score (Piotroski, 2000)")
    st.markdown(
        r"""
Nine binary signals. $F=\sum_{i=1}^{9} F_i \in \{0,\ldots,9\}$.

**Profitability**
- $F_{ROA}=1$ iff $ROA_t = NI_t / \overline{A}_t > 0$
- $F_{CFO}=1$ iff $CFO_t > 0$
- $F_{\Delta ROA}=1$ iff $ROA_t - ROA_{t-1} > 0$
- $F_{ACCRUAL}=1$ iff $CFO_t > NI_t$

**Leverage, liquidity, dilution**
- $F_{\Delta LEVER}=1$ iff $\frac{LTD_t}{A_t} < \frac{LTD_{t-1}}{A_{t-1}}$
- $F_{\Delta LIQUID}=1$ iff $\frac{CA_t}{CL_t} > \frac{CA_{t-1}}{CL_{t-1}}$
- $F_{EQ}=1$ iff $Shares_t \le Shares_{t-1}$

**Efficiency**
- $F_{\Delta MARGIN}=1$ iff $\frac{GP_t}{S_t} > \frac{GP_{t-1}}{S_{t-1}}$
- $F_{\Delta TURN}=1$ iff $\frac{S_t}{\overline{A}_t} > \frac{S_{t-1}}{\overline{A}_{t-1}}$

Interpretation used here: 8–9 high quality, 5–7 average, 0–4 low quality.
        """
    )
    st.subheader("Altman Z-Score (Altman, 1968)")
    st.markdown(
        r"""
$$
Z = 1.2 X_1 + 1.4 X_2 + 3.3 X_3 + 0.6 X_4 + 1.0 X_5
$$

| Factor | Definition | Weight |
|---|---|---|
| $X_1$ | Working capital / Total assets | 1.2 |
| $X_2$ | Retained earnings / Total assets | 1.4 |
| $X_3$ | EBIT / Total assets | 3.3 |
| $X_4$ | Market value of equity / Total liabilities | 0.6 |
| $X_5$ | Sales / Total assets | 1.0 |

Original manufacturing cutoffs: $Z>2.99$ safe, $1.81\le Z\le 2.99$ grey, $Z<1.81$ distress.

This desk maps EBIT to US-GAAP `OperatingIncomeLoss` when a dedicated EBIT tag is absent.
$X_4$ uses Yahoo Finance market cap when available, else price × XBRL shares.
        """
    )

with tab_charts:
    st.subheader("Evolución de los indicadores")
    st.caption("Últimos 5 años fiscales, del más antiguo al más reciente.")

    pc = piotroski_chart(hist_f)
    if pc is not None:
        st.altair_chart(pc, width="stretch")
    else:
        st.info("No hay datos suficientes para graficar Piotroski.")

    pcc = piotroski_criteria_chart(hist_f)
    if pcc is not None:
        st.altair_chart(pcc, width="stretch")
        st.caption("Cada segmento es una señal aprobada (1 punto). La altura total es el F-Score.")

    ac = altman_chart(hist_z)
    if ac is not None:
        st.altair_chart(ac, width="stretch")
        st.caption("X4 usa el cierre en la fecha de cada balance × acciones en circulación.")
    else:
        st.info("No hay datos suficientes para graficar Altman.")

    acc = altman_contrib_chart(hist_z)
    if acc is not None:
        st.altair_chart(acc, width="stretch")
        st.caption("Barras: peso × ratio de cada factor (las negativas bajan el Z). "
                   "El rombo ámbar es el Z-Score total.")






