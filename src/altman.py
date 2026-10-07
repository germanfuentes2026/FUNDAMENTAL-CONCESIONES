"""Altman Z-Score (1968 original manufacturing model) with audit."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.sec_loader import FactPoint, series_by_year


@dataclass
class ComponentResult:
    code: str
    name: str
    weight: float
    value: float | None
    contribution: float | None
    formula: str
    latex: str
    explanation: str
    inputs: dict[str, float | int | str | None] = field(default_factory=dict)


@dataclass
class AltmanResult:
    z: float | None
    year: int | None = None
    zone: str = "N/A"
    zone_detail: str = ""
    components: list[ComponentResult] = field(default_factory=list)
    market_cap: float | None = None
    market_cap_source: str | None = None
    missing: list[str] = field(default_factory=list)


def zone_for(z: float | None) -> tuple[str, str]:
    if z is None:
        return "N/A", "Insufficient inputs to compute Z."
    if z > 2.99:
        return "SAFE", "Z > 2.99 — low bankruptcy probability (original 1968 cutoffs)."
    if z >= 1.81:
        return "GREY", "1.81 ≤ Z ≤ 2.99 — indeterminate / grey zone."
    return "DISTRESS", "Z < 1.81 — high bankruptcy probability (original 1968 cutoffs)."


def _v(series: dict[int, FactPoint], year: int | None) -> float | None:
    if year is None:
        return None
    p = series.get(year)
    return None if p is None else p.value


def _src(p: FactPoint | None) -> dict[str, float | int | str | None]:
    if not p:
        return {"value": None, "concept": None, "end": None, "form": None, "accn": None}
    return {
        "value": p.value,
        "concept": p.concept,
        "end": p.end,
        "fy": p.fy,
        "form": p.form,
        "accn": p.accn,
        "unit": p.unit,
        "source": p.source,
    }


def compute_altman(
    bundle: dict[str, list[FactPoint]],
    *,
    market_cap: float | None = None,
    market_price: float | None = None,
    shares_override: float | None = None,
) -> AltmanResult:
    assets = series_by_year(bundle.get("assets") or [])
    ca = series_by_year(bundle.get("current_assets") or [])
    cl = series_by_year(bundle.get("current_liabilities") or [])
    re = series_by_year(bundle.get("retained_earnings") or [])
    ebit = series_by_year(bundle.get("ebit") or [])
    liab = series_by_year(bundle.get("liabilities") or [])
    rev = series_by_year(bundle.get("revenue") or [])
    shares = series_by_year(bundle.get("shares") or [])

    years = sorted(set(assets.keys()), reverse=True)
    if not years:
        return AltmanResult(z=None, missing=["Total assets not found in XBRL."])
    y = years[0]

    a = _v(assets, y)
    ca_v = _v(ca, y)
    cl_v = _v(cl, y)
    re_v = _v(re, y)
    ebit_v = _v(ebit, y)
    tl = _v(liab, y)
    s = _v(rev, y)
    sh = shares_override if shares_override is not None else _v(shares, y)

    wc = None if ca_v is None or cl_v is None else ca_v - cl_v

    mkt = market_cap
    mkt_src = None
    if mkt is None and market_price is not None and sh is not None:
        mkt = market_price * sh
        mkt_src = "Price × XBRL shares outstanding"
    elif mkt is not None:
        mkt_src = "Market snapshot (Yahoo Finance)"

    def ratio(num: float | None, den: float | None) -> float | None:
        if num is None or den is None or den == 0:
            return None
        return num / den

    x1 = ratio(wc, a)
    x2 = ratio(re_v, a)
    x3 = ratio(ebit_v, a)
    x4 = ratio(mkt, tl)
    x5 = ratio(s, a)

    weights = {"X1": 1.2, "X2": 1.4, "X3": 3.3, "X4": 0.6, "X5": 1.0}
    values = {"X1": x1, "X2": x2, "X3": x3, "X4": x4, "X5": x5}

    missing: list[str] = [k for k, v in values.items() if v is None]
    contribs = {k: (None if values[k] is None else weights[k] * values[k]) for k in weights}
    z = None if missing else sum(contribs[k] or 0.0 for k in weights)
    zone, detail = zone_for(z)

    a_pt = assets.get(y)
    components = [
        ComponentResult(
            code="X1",
            name="Working capital / Total assets",
            weight=1.2,
            value=x1,
            contribution=contribs["X1"],
            formula="X1 = (Current assets − Current liabilities) / Total assets",
            latex=r"X_1=\frac{CA-CL}{A}",
            explanation="Liquidity relative to the asset base.",
            inputs={
                "CA": ca_v,
                "CL": cl_v,
                "WC": wc,
                "A": a,
                **{f"CA.{k}": v for k, v in _src(ca.get(y)).items()},
                **{f"CL.{k}": v for k, v in _src(cl.get(y)).items()},
            },
        ),
        ComponentResult(
            code="X2",
            name="Retained earnings / Total assets",
            weight=1.4,
            value=x2,
            contribution=contribs["X2"],
            formula="X2 = Retained earnings / Total assets",
            latex=r"X_2=\frac{RE}{A}",
            explanation="Cumulative profitability / age and leverage of earnings.",
            inputs={"RE": re_v, "A": a, **{f"RE.{k}": v for k, v in _src(re.get(y)).items()}},
        ),
        ComponentResult(
            code="X3",
            name="EBIT / Total assets",
            weight=3.3,
            value=x3,
            contribution=contribs["X3"],
            formula="X3 = EBIT (Operating income) / Total assets",
            latex=r"X_3=\frac{EBIT}{A}",
            explanation="Operating return on assets, independent of tax and leverage.",
            inputs={"EBIT": ebit_v, "A": a, **{f"EBIT.{k}": v for k, v in _src(ebit.get(y)).items()}},
        ),
        ComponentResult(
            code="X4",
            name="Market value of equity / Total liabilities",
            weight=0.6,
            value=x4,
            contribution=contribs["X4"],
            formula="X4 = Market cap / Total liabilities",
            latex=r"X_4=\frac{MVE}{TL}",
            explanation="Solvency via market cushion over book liabilities.",
            inputs={
                "MVE": mkt,
                "TL": tl,
                "price": market_price,
                "shares": sh,
                "MVE_source": mkt_src,
                **{f"TL.{k}": v for k, v in _src(liab.get(y)).items()},
            },
        ),
        ComponentResult(
            code="X5",
            name="Sales / Total assets",
            weight=1.0,
            value=x5,
            contribution=contribs["X5"],
            formula="X5 = Revenue / Total assets",
            latex=r"X_5=\frac{S}{A}",
            explanation="Asset turnover / capital intensity.",
            inputs={"Sales": s, "A": a, **{f"Sales.{k}": v for k, v in _src(rev.get(y)).items()}},
        ),
    ]

    return AltmanResult(
        z=z,
        year=y,
        zone=zone,
        zone_detail=detail,
        components=components,
        market_cap=mkt,
        market_cap_source=mkt_src,
        missing=missing,
    )
